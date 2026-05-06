# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import contextlib
import math
import os
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.utils.rnn import pad_sequence
import sys
import importlib
from einops import rearrange
from verl import DataProto
from verl.utils.torch_functional import get_eos_mask
import verl.utils.torch_functional as verl_F
from .base import BaseRollout

from transformers import GenerationConfig, AutoProcessor
from transformers.cache_utils import DynamicCache
import torch.nn.functional as F

from PIL import Image
from torchvision.transforms import functional as TVF

from verl.utils.libero_utils import save_rollout_video

try:
    from verl.utils.libero_utils import (
        get_libero_env, get_libero_dummy_action, get_libero_image, 
        get_libero_wrist_image, quat2axisangle, normalize_gripper_action, 
        invert_gripper_action
    )
except ImportError as e:
    print(f"Warning : can't import libero: {e}")
    
from verl.utils.vla_utils.openvla_oft.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
)

import numpy as np
from PIL import Image
# import tensorflow as tf
from collections import deque
import random
import yaml
from pathlib import Path
torch.set_printoptions(threshold=10_0000000)

import threading
import queue
import gc
from collections import defaultdict
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from codetiming import Timer

# For Libero multiprocessing
import multiprocessing
from multiprocessing import Process, Queue

__all__ = ['RobHFRollout']

# Environment initialization lock for Robotwin
_ENV_INIT_LOCK = threading.Lock()

OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def center_crop_image(img):
    width, height = img.size
    crop_scale = 0.9
    crop_h = int(height * crop_scale)
    crop_w = int(width * crop_scale)
    img = TVF.center_crop(img, [crop_h, crop_w])
    img = TVF.resize(img, [height, width])
    return img


# ================ Robotwin-specific functions ================

def normalize_proprio(proprio, norm_stats):
    """Normalize proprioception data for Robotwin."""
    if ACTION_PROPRIO_NORMALIZATION_TYPE == "bounds":
        mask = norm_stats.get("mask", np.ones_like(norm_stats["min"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["max"]), np.array(norm_stats["min"])
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == "bounds_q99":
        mask = norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
    else:
        raise ValueError("Unsupported action/proprio normalization type detected!")
    
    normalized_proprio = np.clip(
        np.where(
            mask,
            2 * (proprio - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1,
            proprio,
        ),
        a_min=-1.0,
        a_max=1.0,
    )
    return normalized_proprio



def get_robotwin2_task(task_name, config):
    """Get robotwin 2.0 task"""
    robotwin2_path = os.path.join(os.path.dirname(__file__), '..', '..', 'utils', 'envs', 'robotwin2')
    if robotwin2_path not in sys.path:
        sys.path.append(robotwin2_path)
        
    robotwin2_utils_path = os.path.join(os.path.dirname(__file__), '..', '..', 'utils', 'envs', 'robotwin2', "description", "utils")
    if robotwin2_utils_path not in sys.path:
        sys.path.append(robotwin2_utils_path)
    
    from envs import CONFIGS_PATH
    
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit(f"No Task: {task_name}")
    
    task_config = config.get('twin2_task_config', 'demo_randomized')
    config_file = os.path.join(robotwin2_path, f"task_config/{task_config}.yml")
    
    with open(config_file, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    args['task_name'] = task_name
    args['task_config'] = task_config
    args['ckpt_setting'] = config.get('twin2_ckpt_setting', 'demo_randomized')
    
    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise ValueError("No embodiment files")
        return robot_file
    
    def get_embodiment_config(robot_file):
        robot_config_file = os.path.join(robot_file, "config.yml")
        with open(robot_config_file, "r", encoding="utf-8") as f:
            embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
        return embodiment_args
    
    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment items should be 1 or 3")
    
    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    
    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]
    
    args["eval_mode"] = True
    args["eval_video_log"] = False
    args["render_freq"] = 0
    args['instruction_type'] = config.get('twin2_instruction_type', 'unseen')
    
    return env_instance, args

def encode_obs(observation):
    """Post-Process Observation for robotwin 2.0"""
    return observation

class RobotwinEnvWrapper:
    """Thread-safe wrapper for Robotwin environment (supports both 1.0 and 2.0)"""
    def __init__(self, task_name, trial_id, trial_seed, config, version="1.0"):
        self.task_name = task_name
        self.trial_id = trial_id
        self.trial_seed = trial_seed
        self.config = config
        self.version = version
        self.env = None
        self.args = None
        self.active = True
        self.complete = False
        self.finish_step = 0
        self.lock = threading.Lock()
        self.instruction = None
        
    def initialize(self):
        """Initialize the environment"""
        with _ENV_INIT_LOCK:
            with self.lock:
                try:
                    if self.version == "1.0":
                        print("RobotWin 2.0 fully encompasses RobotWin 1.0, therefore we prioritize support for RobotWin 2.0")
                        raise ValueError
                    else:  # 2.0
                        self.env, self.args = get_robotwin2_task(self.task_name, self.config)
                        self.env.setup_demo(now_ep_num=self.trial_id, seed=self.trial_seed, is_test=True, **self.args)
                        episode_info_list = [self.env.get_info()]
                except Exception as e:
                    print(f"****** IN thread: setup_demo ERROR {e} ******", flush=True)
                    torch.cuda.empty_cache()
                    gc.collect()
                    self.env, self.args = get_robotwin2_task(self.task_name, self.config)
                    self.env.setup_demo(now_ep_num=self.trial_id, seed=self.trial_seed, is_test=True, **self.args)
                    episode_info_list = [self.env.get_info()]
                
                
                from generate_episode_instructions import generate_episode_descriptions
                results = generate_episode_descriptions(self.task_name, episode_info_list, 1, seed=self.trial_id)
                self.instruction = np.random.choice(results[0][self.args["instruction_type"]])
                self.env.set_instruction(instruction=self.instruction)
                
    def get_obs(self):
        """Get observation from environment"""
        with self.lock:
            try:
                geted_obs = self.env.get_obs()
                return geted_obs
            except Exception as e:
                print(f"****** IN thread: get_obs ERROR {e} ******", flush=True)
                torch.cuda.empty_cache()
                gc.collect()
                geted_obs = self.env.get_obs()
                return geted_obs
    
    def get_instruction(self):
        """Get instruction for the task"""
        with self.lock:
            
            return self.env.get_instruction()
            
    def step(self, action):
        """Execute action in environment"""
        with self.lock:
            try:
                
                self.env.take_action(action)
                done = self.env.eval_success
                    
            except Exception as e:
                done = False
                error_msg = f"****** action execution ERROR: {type(e).__name__}: {str(e)} ******"
                print(error_msg, flush=True)
                traceback.print_exc()
                
            try:
                obs = self.env.get_obs()
                obs = encode_obs(obs)
            except Exception as e:
                print(f"****** env.get_obs ERROR {e} ******", flush=True)
                obs = None
                
            self.finish_step += action.shape[0]
            
            if done or self.finish_step >= self.env.step_lim:
                self.active = False
                self.complete = done
            
            return obs, done
            
    def close(self):
        """Close the environment"""
        with self.lock:
            if self.env is not None:
                try:
                    self.env.close_env(clear_cache=True)
                except Exception as e:
                    print(f"******IN env.close ERROR {e} ******", flush=True)

# ================ Libero-specific functions ================

def env_worker(task_name, task_id, trial_id, config, input_queue, output_queue, is_valid, global_steps, max_steps):
    """Worker process for Libero environments."""

    from libero.libero import benchmark
    
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_name]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    initial_state = initial_states[trial_id]

    if task_name == "libero_spatial" and task_id == 5:
        initial_state[12] += 0.038
    
    env = None
    while True:
        # try:
        env, task_description = get_libero_env(task, config.model_family, resolution=256)
        break
        # except:
        #     print(f"*** env initialization failed ***")
        #     if env is not None:
        #         try:
        #             env.close()
        #         except Exception as e:
        #             print(f"error when close the env: {e}")
        #     torch.cuda.empty_cache()
        #     gc.collect()
        #     print("gc collect finish")
    
    env.reset()
    obs = env.set_init_state(initial_state)
    
    t = 0
    valid_images = []
    while t < config.num_steps_wait:
        obs, _, _, _ = env.step(get_libero_dummy_action(config.model_family))
        t += 1
        
    if is_valid:
        img = obs["agentview_image"][::-1, ::-1]
        valid_images.append(img)
    
    output_queue.put({
        'type': 'init',
        'obs': obs,
        "task_description": task_description,
        'valid_images': valid_images.copy(),
        'task_file_name': f"{task_name}_task_{task_id}_trial_{trial_id}",
        'active': True,
        'complete': False,
        'finish_step': 0
    })
    
    active = True
    complete = False
    finish_step = 0
    
    while True:
        action = input_queue.get()
        if action is None:
            env.close()
            output_queue.put({'type': 'terminate'})
            break
        
        step_images = []
        for i in range(len(action)):
            a = action[i]
            normalized_action = normalize_gripper_action(a, binarize=True)
            inverted_action = invert_gripper_action(normalized_action)
            obs, reward, done, info = env.step(inverted_action.tolist())
            
            if is_valid:
                img = obs["agentview_image"][::-1, ::-1]
                step_images.append(img)
            
            finish_step += 1
            if done or finish_step >= max_steps:
                active = False
                complete = done
                break
        
        output_data = {
            'type': 'step',
            'obs': obs,
            'active': active,
            'complete': complete,
            'finish_step': finish_step,
            'valid_images': step_images.copy() if is_valid else []
        }
        output_queue.put(output_data)

# ================ Main Rollout Class ================

from dataclasses import dataclass
from verl.workers.actor import ActionTokenizer

import json

class RobHFRollout(BaseRollout):
    def __init__(self, module: nn.Module, config):
        super().__init__()
        self.config = config
        self.module = module
        self.max_steps = {
            "libero_spatial": self.config.libero_spatial_max_steps,
            "libero_object": self.config.libero_object_max_steps,
            "libero_goal": self.config.libero_goal_max_steps,
            "libero_10": self.config.libero_10_max_steps,
            "libero_90": 512,
            "robotwin2_click_bell": 200,
            "robotwin2_move_can_pot": 200,
            "robotwin2_place_phone_stand": 200,
            "robotwin2_place_a2b_left": 200,
            "robotwin2_place_a2b_right": 200,
            "robotwin2_handover_mic": 200,
            "robotwin2_pick_dual_bottles": 100,
            "robotwin2_lift_pot": 200,
            "robotwin2_put_bottles_dustbin": 800,
            "robotwin2_stack_blocks_two": 400,
            "robotwin2_stack_bowls_two": 400,
            "robotwin2_handover_block": 400,
            "robotwin2_place_empty_cup": 200,
            "robotwin2_shake_bottle": 75,
            "robotwin2_move_stapler_pad": 200,
            "robotwin2_place_container_plate": 150,
            "robotwin2_blocks_ranking_rgb": 600,
            "robotwin2_beat_block_hammer": 200,
            "robotwin2_place_mouse_pad": 200,
            "robotwin2_place_shoe": 250,
            "robotwin2_move_pillbottle_pad": 200,
        }

        self.processor = AutoProcessor.from_pretrained(config.pretrained_checkpoint, local_files_only=True)
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer, need_to_sub=self.config.need_to_sub)

        self.action_0_id = self.processor.tokenizer.vocab.get("<action_0>")
        self.latent_start_id = self.processor.tokenizer.vocab.get("<latent_start>")
        self.latent_end_id = self.processor.tokenizer.vocab.get("<latent_end>")
        self.latent_pad_id = self.processor.tokenizer.vocab.get("<latent_pad>")

        self.latent_end_num = getattr(self.config, 'latent_end_num', 4)
        self.input_mode = getattr(self.config, 'input_mode', 'embeds')
        self.latent_bind = getattr(self.config, 'latent_bind', 0)
        if self.latent_end_num > 1 and getattr(self.config, 'latent_length', 8) > 0:
            self.G = self.config.latent_length // self.latent_end_num
        else:
            self.G = 0

        statistics_path = self.config.data_status
        with open(statistics_path, 'r') as f:
            self.stats_data = json.load(f)

        self.dataset_name = next(iter(self.stats_data))
        self.action_mask = np.array(self.stats_data[self.dataset_name]['action']['mask'])
        self.action_min = np.array(self.stats_data[self.dataset_name]['action']['q01'])
        self.action_max = np.array(self.stats_data[self.dataset_name]['action']['q99'])
        self.state_mask = np.array(self.stats_data[self.dataset_name]['state']['mask'])
        self.state_min = np.array(self.stats_data[self.dataset_name]['state']['q01'])
        self.state_max = np.array(self.stats_data[self.dataset_name]['state']['q99'])
        
        self.vla_preprocess()
        
        # Setup execution pool based on task suite
        if "robotwin" in self.config.task_suite_name:
            self.env_thread_pool = ThreadPoolExecutor(max_workers=16)
            self.robotwin_version = self._detect_robotwin_version()
        
    def _detect_robotwin_version(self):
        """Detect which version of robotwin to use based on config"""
        if hasattr(self.config, 'robotwin_version'):
            return self.config.robotwin_version
        elif 'robotwin2' in self.config.task_suite_name:
            return "2.0"
        else:
            print("RobotWin 2.0 fully encompasses RobotWin 1.0, therefore we prioritize support for RobotWin 2.0")
            raise ValueError
        
    def vla_preprocess(self):
        # if self.config.vla in ["openvla", "openvla-oft", "janus-oft", "qwen-oft"]:
        #     gpus = tf.config.experimental.list_physical_devices('GPU')
        #     if gpus:
        #         for gpu in gpus:
        #             tf.config.experimental.set_memory_growth(gpu, True)
        
        if self.config.vla in ["openvla-oft"]:
            if "libero" in self.config.task_suite_name:
                if self.config.unnorm_key not in self.module.norm_stats and f"{self.config.unnorm_key}_no_noops" in self.module.norm_stats:
                    self.config.unnorm_key = f"{self.config.unnorm_key}_no_noops"
            elif "robotwin" in self.config.task_suite_name:
                self.config.unnorm_key = self.config.unnorm_key.removeprefix("robotwin_").removeprefix("robotwin2_")
            assert self.config.unnorm_key in self.module.norm_stats, f"Action un-norm key {self.config.unnorm_key} not found in VLA `norm_stats`!"


    def generate_sequences(self, prompts):
        batch_size = prompts.batch.batch_size[0]
        
        if prompts.meta_info.get('n_samples') is None:
            micro_batch_size = self.config.val_micro_batch_size if self.config.val_micro_batch_size is not None else 1
        else:
            micro_batch_size = self.config.get('micro_batch_size', batch_size)

        num_chunks = max(batch_size // micro_batch_size, 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        output = [self._generate_minibatch(p) for p in batch_prompts]
        output = DataProto.concat(output)
        return output

    def process_input(self, inputs: list, task_descriptions: list):
        batch_size = len(inputs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pad_id = self.processor.tokenizer.pad_token_id

        input_ids_list = []
        encoder_pixel_values_list = []
        image_grid_thw_list = []

        for i in range(batch_size):
            input_data = inputs[i]
            task_description = task_descriptions[i]

            images = []
            image = Image.fromarray(input_data["full_image"]).convert("RGB")
            if getattr(self.config, "center_crop", False):
                image = center_crop_image(image)
            images.append(image)

            if getattr(self.config, "num_images_in_input", 1) > 1:
                if "robotwin" in self.config.task_suite_name.lower():
                    for key in input_data:
                        if "wrist" in key and isinstance(input_data[key], np.ndarray):
                            wrist_image = Image.fromarray(input_data[key]).convert("RGB")
                            if getattr(self.config, "center_crop", False):
                                wrist_image = center_crop_image(wrist_image)
                            images.append(wrist_image)
                else:  # Libero
                    if "wrist_image" in input_data:
                        wrist_image = Image.fromarray(input_data["wrist_image"]).convert("RGB")
                        if getattr(self.config, "center_crop", False):
                            wrist_image = center_crop_image(wrist_image)
                        images.append(wrist_image)

            state_tokens = ""
            if getattr(self.config, "use_proprio", False):
                state = input_data["state"]
                normalized_state = np.where(
                    self.state_mask,
                    np.clip(2 * (state - self.state_min) / (self.state_max - self.state_min + 1e-8) - 1, -1, 1),
                    state
                )
                state_tokens += self.action_tokenizer(normalized_state)


            prompt_content = []
            for image in images:
                prompt_content.append({"type": "image", "image": image})
            prompt_content.append({"type": "text", "text": task_description + state_tokens})

            messages = [
                {
                    "role": "user",
                    "content": prompt_content,
                }
            ]

            inputs_prepare = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )

            ids = inputs_prepare.input_ids[0].to(device)
            input_ids_list.append(ids)
            encoder_pixel_values_list.append(inputs_prepare.pixel_values.to(device))
            image_grid_thw_list.append(inputs_prepare.image_grid_thw[0].to(device))

        # --- Left-pad the image+text prefix to a uniform length BEFORE appending latent/action tail ---
        # This ensures every sample's prompt occupies the same positions, so the fixed tail
        # (latent body + action placeholders) always starts at the same offset in the batch tensor.
        max_prompt_len = max(ids.size(0) for ids in input_ids_list)
        max_prompt_len = max(max_prompt_len, self.config.max_prompt_length)

        if getattr(self.config, "use_latent", False) and self.latent_end_num > 1 and self.latent_bind == 0:
            action_token_num = self.config.action_chunks_len * self.config.action_token_len
            group_tokens = [self.latent_pad_id] * self.G + [self.latent_end_id]
            body_tokens = group_tokens * self.latent_end_num
            tail_tokens = [self.latent_start_id] + body_tokens + [self.action_0_id] * action_token_num
            tail_tensor = torch.tensor(tail_tokens, dtype=torch.long, device=device)
            tail_len = tail_tensor.size(0)
            total_len = max_prompt_len + tail_len
        else:
            tail_tensor = None
            tail_len = 0
            total_len = max_prompt_len
            if getattr(self.config, "use_latent", False):
                # latent_start only (bind=1 or end_num=1 paths)
                total_len += 1

        batch_input_ids = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros(batch_size, total_len, device=device, dtype=torch.long)

        for i, ids in enumerate(input_ids_list):
            l = ids.size(0)
            # left-pad the prompt into [0 : max_prompt_len]
            batch_input_ids[i, max_prompt_len - l : max_prompt_len] = ids
            attention_mask[i, max_prompt_len - l : max_prompt_len] = 1
            # append tail (latent+action) at fixed positions [max_prompt_len : total_len]
            if tail_tensor is not None:
                batch_input_ids[i, max_prompt_len:] = tail_tensor
                attention_mask[i, max_prompt_len:] = 1
            elif getattr(self.config, "use_latent", False):
                batch_input_ids[i, max_prompt_len] = self.latent_start_id
                attention_mask[i, max_prompt_len] = 1

        batch_pixel_values = torch.stack([px.to(torch.bfloat16) for px in encoder_pixel_values_list], dim=0)
        batch_image_grid_thw = torch.stack([ig.to(device) for ig in image_grid_thw_list], dim=0)

        batchdata = {
            "input_ids": batch_input_ids,
            "pixel_values": batch_pixel_values,
            "attention_mask": attention_mask,
            "image_grid_thw": batch_image_grid_thw,
        }

        return batchdata
        
    def _generate_minibatch(self, prompts):
        """Generate minibatch - routes to appropriate implementation based on task suite"""
        if "robotwin" in self.config.task_suite_name:
            return self._generate_minibatch_robotwin(prompts)
        else:
            return self._generate_minibatch_libero(prompts)
    
    def _generate_minibatch_robotwin(self, prompts):
        """Generate minibatch for Robotwin using threading"""
        self.module.eval()
        meta_info = prompts.meta_info
        n_samples = meta_info.get('n_samples', 1)
        task_id = prompts.batch['task_id'].repeat_interleave(n_samples, dim=0)
        trial_id = prompts.batch['trial_id'].repeat_interleave(n_samples, dim=0)
        trial_seed = prompts.batch['trial_seed'].repeat_interleave(n_samples, dim=0)
        task_suite_name = np.repeat(prompts.non_tensor_batch['task_suite_name'], n_samples)
        max_steps = self.max_steps.get(self.config.task_suite_name, 800)
        batch_size = task_id.size(0)
        is_valid = meta_info.get('n_samples') is None
        global_steps = meta_info.get('global_steps', 0) if is_valid else 0
        
        # Create environment wrappers
        env_wrappers = []
        for idx in range(batch_size):
            task_name = task_suite_name[idx].removeprefix("robotwin_").removeprefix("robotwin2_")
            t_id = task_id[idx][0].item()
            tr_id = trial_id[idx][0].item()
            tr_seed = trial_seed[idx][0].item()
            
            wrapper = RobotwinEnvWrapper(task_name, tr_id, tr_seed, self.config, version=self.robotwin_version)
            env_wrappers.append(wrapper)
        
        # Initialize environments in parallel
        init_futures = []
        for wrapper in env_wrappers:
            future = self.env_thread_pool.submit(wrapper.initialize)
            init_futures.append(future)
        
        for future in as_completed(init_futures, timeout=360):
            try:
                future.result()
            except Exception as e:
                print(f"Environment initialization failed: {e}", flush=True)
                traceback.print_exc()
                raise
        
        # Collect initial observations
        inputs = []
        task_descriptions = []
        task_records = []
        valid_video = defaultdict(list)
        
        for idx, wrapper in enumerate(env_wrappers):
            try:
                obs = wrapper.get_obs()
                obs = encode_obs(obs)
                    
                task_description = wrapper.get_instruction()
                task_descriptions.append(task_description)
                inputs.append(self._obs_to_input(obs, is_robotwin=True, robotwin_version=wrapper.version))
                
                task_file_name = f"{wrapper.task_name}_trial_{wrapper.trial_id}_seed_{wrapper.trial_seed}"
                task_records.append({
                    "active": wrapper.active,
                    "complete": wrapper.complete,
                    "finish_step": wrapper.finish_step,
                    "task_file_name": task_file_name
                })
                
                if is_valid:
                    img = obs['observation']['head_camera']['rgb']
                    valid_video[task_file_name].append(img)
                    
            except Exception as e:
                print(f"Failed to get initial observation: {e}", flush=True)
                traceback.print_exc()
                raise
        
        # Main rollout loop
        step = 0
        vla_history = []
        
        while step < max_steps:
            active_indices = [i for i, r in enumerate(task_records) if r['active']]
            
            current_inputs = inputs
            current_task_descriptions = task_descriptions
            
            # Get VLA actions
            vla_input = self.process_input(current_inputs, current_task_descriptions)
            vla_input.update(meta_info)
            
            vla_output = self._generate_one_step(vla_input)
            actions = vla_output["action"]
            
            step_data = {
                "responses": vla_output["responses"],
                "input_ids": vla_output["input_ids"],
                "attention_mask": vla_output["attention_mask"],
                "pixel_values": vla_output["pixel_values"],
                "action": actions,
                "step": step
            }
            if vla_output.get("proprio") is not None:
                step_data["proprio"] = vla_output["proprio"]
                
            vla_history.append(step_data)
            
            # Execute actions in parallel
            step_futures = []
            for idx in active_indices:
                future = self.env_thread_pool.submit(
                    env_wrappers[idx].step,
                    actions[idx]
                )
                step_futures.append((idx, future))
            
            # Collect results
            new_inputs = inputs.copy()
            for idx, future in step_futures:
                try:
                    obs, done = future.result(timeout=120)
                    if obs is not None:
                        obs = encode_obs(obs)
                        new_inputs[idx] = self._obs_to_input(obs, is_robotwin=True, robotwin_version=env_wrappers[idx].version)
                        
                    task_records[idx]['active'] = env_wrappers[idx].active
                    task_records[idx]['complete'] = env_wrappers[idx].complete
                    task_records[idx]['finish_step'] = env_wrappers[idx].finish_step
                    
                    if is_valid and obs is not None:
                        img = obs['observation']['head_camera']['rgb']
                        valid_video[task_records[idx]['task_file_name']].append(img)
                        
                except Exception as e:
                    print(f"Step execution failed: {e}", flush=True)
                    task_records[idx]['active'] = False
                    task_records[idx]['complete'] = False
                    task_records[idx]['finish_step'] = step + self.config.action_chunks_len
            
            inputs = new_inputs
            step += self.config.action_chunks_len
        
        # Clean up environments
        cleanup_futures = []
        for wrapper in env_wrappers:
            future = self.env_thread_pool.submit(wrapper.close)
            cleanup_futures.append(future)
            
        for future in as_completed(cleanup_futures):
            try:
                future.result(timeout=20)
            except Exception as e:
                print(f"Environment cleanup failed: {e}", flush=True)
        
        torch.cuda.empty_cache()
        gc.collect()
        
        # Save validation videos
        if is_valid:
            for task_file, images in valid_video.items():
                complete = any(r['complete'] for r in task_records if r['task_file_name'] == task_file)
                save_rollout_video(
                    images,
                    self.config.experiment_name,
                    task_file,
                    global_steps,
                    complete
                )
        
        self.module.train()
        return self._prepare_output_batch(vla_history, task_records, batch_size)
    
    def _generate_minibatch_libero(self, prompts):
        """Generate minibatch for Libero using multiprocessing"""
        self.module.eval()
        meta_info = prompts.meta_info
        n_samples = meta_info.get('n_samples', 1)
        task_id = prompts.batch['task_id'].repeat_interleave(n_samples, dim=0)
        trial_id = prompts.batch['trial_id'].repeat_interleave(n_samples, dim=0)
        task_suite_name = np.repeat(prompts.non_tensor_batch['task_suite_name'], n_samples)
        max_steps = self.max_steps[self.config.task_suite_name]
        batch_size = task_id.size(0)
        is_valid = meta_info.get('n_samples') is None
        global_steps = meta_info.get('global_steps', 0) if is_valid else 0
        
        processes = []
        input_queues = []
        output_queues = []
        
        for idx in range(batch_size):
            task_name = task_suite_name[idx]
            t_id = task_id[idx][0].item()
            tr_id = trial_id[idx][0].item()
            input_q = Queue()
            output_q = Queue()
            p = Process(
                target=env_worker,
                args=(task_name, t_id, tr_id, self.config, input_q, output_q, is_valid, global_steps, max_steps)
            )
            p.start()
            processes.append(p)
            input_queues.append(input_q)
            output_queues.append(output_q)
        
        inputs = []
        task_descriptions = []
        task_records = []
        valid_video = defaultdict(list)
        
        for idx in range(batch_size):
            init_data = output_queues[idx].get(timeout=120)
            assert init_data['type'] == 'init'
            task_descriptions.append(init_data["task_description"])
            inputs.append(self._obs_to_input(init_data['obs'], is_robotwin=False))
            task_records.append({
                "active": init_data['active'],
                "complete": init_data['complete'],
                "finish_step": init_data['finish_step'],
                "task_file_name": init_data['task_file_name']
            })
            if is_valid:
                valid_video[init_data['task_file_name']].extend(init_data['valid_images'])
        
        step = 0
        vla_history = []
        
        while step < max_steps:
            active_indices = [i for i, r in enumerate(task_records) if r['active']]
            
            current_inputs = inputs
            current_task_descriptions = task_descriptions
            
            vla_input = self.process_input(current_inputs, current_task_descriptions)
            vla_input.update(meta_info)

            vla_output = self._generate_one_step(vla_input)
            actions = vla_output["action"]
            
            step_data = {
                "responses": vla_output["responses"],
                "input_ids": vla_output["input_ids"],
                "attention_mask": vla_output["attention_mask"],
                "pixel_values": vla_output["pixel_values"],
                "action": actions,
                "old_log_probs": vla_output["old_log_probs"],
                "image_grid_thw": vla_output["image_grid_thw"],
                "step": step,
                "old_values": vla_output["old_values"],
                "old_latents": vla_output["old_latents"],
                "old_latent_end_log_prob": vla_output["old_latent_end_log_prob"],
                "latent_mask": vla_output.get("latent_mask"),
                "chosen_length": vla_output.get("chosen_length"),
            }
            vla_history.append(step_data)
            
            for idx in active_indices:
                input_queues[idx].put(actions[idx])
            
            new_inputs = inputs.copy()
            for idx in active_indices:
                result = output_queues[idx].get(timeout=30)
                assert result['type'] == 'step'
                new_inputs[idx] = self._obs_to_input(result['obs'], is_robotwin=False)
                task_records[idx]['active'] = result['active']
                task_records[idx]['complete'] = result['complete']
                task_records[idx]['finish_step'] = result['finish_step']
                if is_valid:
                    valid_video[task_records[idx]['task_file_name']].extend(result['valid_images'])
            
            inputs = new_inputs
            step += self.config.action_chunks_len
        
        if self.config.bootstrap != 'none':
            vla_input = self.process_input(inputs, task_descriptions)
            vla_input.update(meta_info)
            vla_output = self._generate_one_step(vla_input)
            step_data = {
                "old_values": vla_output["old_values"],
            }
            vla_history.append(step_data)
        
        for q in input_queues:
            q.put(None)
        for p in processes:
            p.join(timeout=20)
            if p.is_alive():
                p.terminate()
        
        torch.cuda.empty_cache()
        
        if is_valid:
            for task_file, images in valid_video.items():
                complete = any(r['complete'] for r in task_records if r['task_file_name'] == task_file)
                save_rollout_video(
                    images,
                    self.config.experiment_name,
                    task_file,
                    global_steps,
                    complete
                )
        
        self.module.train()
        
        return self._prepare_output_batch(vla_history, task_records, batch_size)
    
    def _prepare_output_batch(self, vla_history, task_records, batch_size):
        """Prepare the output batch from VLA history.
        When bootstrap != 'none', the last entry in vla_history only has 'old_values' (bootstrap value),
        so we use vla_history[:-1] for other keys and full vla_history for old_values.
        """
        batch = {
            'responses': [],
            'input_ids': [],
            'attention_mask': [],
            'pixel_values': [],
            'old_log_probs': [],
            'image_grid_thw': [],
            'old_values': [],
        }
        
        key_names = ["responses", "input_ids", "attention_mask", "pixel_values", "old_log_probs", "image_grid_thw", "old_values"]
        history_main = vla_history[:-1] if getattr(self.config, 'bootstrap', 'none') != 'none' and len(vla_history) > 1 else vla_history
        history_values = vla_history


        has_old_latents = ( 
            len(history_main) > 0 and 
            "old_latents" in history_main[0] and 
            history_main[0]["old_latents"] is not None
        )
        if has_old_latents:
            batch['old_latents'] = []
            key_names.append("old_latents")

        has_latent_mask = (
            len(history_main) > 0 and
            "latent_mask" in history_main[0] and
            history_main[0]["latent_mask"] is not None
        )
        if has_latent_mask:
            batch['latent_mask'] = []
            key_names.append("latent_mask")

        has_chosen_length = (
            len(history_main) > 0 and
            "chosen_length" in history_main[0] and
            history_main[0]["chosen_length"] is not None
        )
        if has_chosen_length:
            batch['chosen_length'] = []
            key_names.append("chosen_length")

        has_latent_end_log_prob = (
            len(history_main) > 0 and
            "old_latent_end_log_prob" in history_main[0] and
            history_main[0]["old_latent_end_log_prob"] is not None
        )
        if has_latent_end_log_prob:
            batch['old_latent_end_log_prob'] = []
            key_names.append("old_latent_end_log_prob")

        for k in key_names:
            hist = history_values if k == 'old_values' else history_main
            for h in hist:
                batch[k].append(h[k])

        for k, v in batch.items():
            batch[k] = torch.stack(v, dim=1)
        
        batch["complete"] = torch.tensor([bool(k["complete"]) for k in task_records], dtype=torch.bool, device=batch['responses'].device)
        batch["finish_step"] = torch.tensor([k["finish_step"] for k in task_records], dtype=torch.int64, device=batch['responses'].device)
        
        output_batch = TensorDict(batch, batch_size=batch_size)
        return DataProto(batch=output_batch)
    

    @torch.no_grad()
    def _generate_one_step(self, prompts: dict):
        """Generate one step of actions"""
        if self.config.vla == "qwen-oft":
            return self._generate_one_step_qwen_oft(prompts)
        else:
            raise ValueError(f"Unknown VLA type: {self.config.vla}")


    def _generate_one_step_qwen_oft(self, prompts: dict):
        """Generate one step for Qwen-OFT with grouped AR latent (bind=0, latent_end_num=4)."""
        idx = prompts['input_ids']
        attention_mask = prompts['attention_mask']
        pixel_values = prompts["pixel_values"]
        image_grid_thw = prompts["image_grid_thw"]
        
        do_sample = prompts.get('do_sample', self.config.do_sample)
        temperature = prompts.get('temperature', self.config.temperature)
        action_token_num = self.config.action_chunks_len * self.config.action_token_len
        
        param_ctx = contextlib.nullcontext()
        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        
        with param_ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                device = idx.device
                B = idx.shape[0]
                L = self.config.latent_length
                latent_end_num = self.latent_end_num
                latent_bind = self.latent_bind
                G = self.G
                max_groups = latent_end_num
                input_mode = self.input_mode

                if self.config.use_latent and latent_end_num > 1 and latent_bind == 0:
                    body_len = max_groups * (G + 1)  # 12
                    prompt_start_len = idx.shape[1] - body_len - action_token_num

                    # Phase 1: Prefill (prompt + latent_start)
                    prefill_mask = attention_mask[:, :prompt_start_len]
                    if input_mode == "ids":
                        prefill_ids = idx[:, :prompt_start_len]
                        outputs_prefill = self.module.model(
                            input_ids=prefill_ids,
                            pixel_values=pixel_values,
                            attention_mask=prefill_mask,
                            image_grid_thw=image_grid_thw,
                            return_dict=True, action_length=0, use_cache=True,
                            attn_mode=self.config.attn_mode,
                        )
                    else:
                        inputs_embeds = self.module.language_model.get_input_embeddings()(idx)
                        prefill_embeds = inputs_embeds[:, :prompt_start_len, :]
                        outputs_prefill = self.module.model(
                            inputs_embeds=prefill_embeds,
                            pixel_values=pixel_values,
                            attention_mask=prefill_mask,
                            image_grid_thw=image_grid_thw,
                            return_dict=True, action_length=0, use_cache=True,
                            attn_mode=self.config.attn_mode,
                        )

                    past_key_values = outputs_prefill.past_key_values
                    prefill_len = past_key_values.get_seq_length()
                    past_length = prefill_len
                    prev_latent_hidden = outputs_prefill.last_hidden_state[:, -1, :]
                    D = prev_latent_hidden.shape[-1]

                    latent_end_embed = self.module.language_model.get_input_embeddings()(
                        torch.tensor([[self.latent_end_id]], device=device, dtype=torch.long)
                    ).to(dtype=prev_latent_hidden.dtype).expand(B, 1, -1)

                    # Phase 2: Grouped AR Loop
                    collected_latents = []
                    end_logits = []
                    cur_attention = prefill_mask.clone()
                    for_value_hidden_states = []

                    collected_latents.append(prev_latent_hidden)

                    def _ar_step(embeds, cur_attention, past_length, past_key_values):
                        cur_attention = torch.cat([
                            cur_attention,
                            torch.ones(B, 1, device=device, dtype=cur_attention.dtype),
                        ], dim=1)
                        cache_position = torch.arange(past_length, past_length + 1, device=device, dtype=torch.long)
                        out = self.module.model(
                            inputs_embeds=embeds,
                            pixel_values=None,
                            image_grid_thw=None,
                            attention_mask=cur_attention,
                            cache_position=cache_position,
                            return_dict=True, action_length=0, use_cache=True,
                            past_key_values=past_key_values,
                            attn_mode=self.config.attn_mode,
                        )
                        past_length += 1
                        past_key_values = out.past_key_values
                        return out.last_hidden_state[:, -1, :], cur_attention, past_length, past_key_values

                    for _g in range(max_groups):
                        for _p in range(G):
                            prev_latent_hidden, cur_attention, past_length, past_key_values = _ar_step(
                                prev_latent_hidden.unsqueeze(1).to(prev_latent_hidden.dtype),
                                cur_attention,
                                past_length,
                                past_key_values
                            )
                            if _p == 0:
                                collected_latents.append(prev_latent_hidden)

                        step_logits = self.module.lm_head(prev_latent_hidden.unsqueeze(1)).squeeze(1)
                        end_logits.append(step_logits)

                        prev_latent_hidden, cur_attention, past_length, past_key_values = _ar_step(latent_end_embed, cur_attention, past_length, past_key_values)
                        collected_latents.append(prev_latent_hidden)
                        for_value_hidden_states.append(prev_latent_hidden)
                        
                    for_value_hidden_states = torch.stack(for_value_hidden_states, dim=1)  # (B, max_groups, D)
                    
                    candidate_sizes = [G * (g + 1) for g in range(max_groups)]
                    group_end_logits = torch.stack(end_logits, dim=1)  
                    if not torch.isfinite(group_end_logits).all():
                        print(
                            "[rob_rollout] Warning: non-finite values detected in group_end_logits; "
                            "replacing nan/inf with dtype min."
                        )
                        min_val = torch.finfo(group_end_logits.dtype).min
                        group_end_logits = torch.nan_to_num(
                            group_end_logits,
                            nan=min_val,
                            posinf=min_val,
                            neginf=min_val,
                        )

                    if do_sample and self.config.end_do_sample:
                        group_end_logits = group_end_logits / temperature
                        group_end_probs = torch.softmax(group_end_logits[:, :, self.latent_end_id], dim=-1)
                        chosen_idx = torch.multinomial(group_end_probs.view(-1, group_end_probs.shape[-1]), num_samples=1).squeeze(-1)
                        group_end_log_probs = F.log_softmax(group_end_logits, dim=-1)[:, :, self.latent_end_id]
                    else:
                        group_end_log_probs = F.log_softmax(group_end_logits, dim=-1)[:, :, self.latent_end_id]
                        if self.config.latent_group_mode == 'max':
                            chosen_idx = group_end_log_probs.argmax(dim=1)
                        elif self.config.latent_group_mode == 'first':
                            threshold_log = math.log(0.99)
                            exceeds = group_end_log_probs >= threshold_log
                            has_any = exceeds.any(dim=1)
                            first_exceed_idx = torch.argmax(exceeds.to(group_end_log_probs.dtype), dim=1)
                            best_group_idx = group_end_log_probs.argmax(dim=1)
                            chosen_idx = torch.where(has_any, first_exceed_idx, best_group_idx)
                        else:
                            raise ValueError(f"Unknown latent group mode: {self.config.latent_group_mode}")


                    candidate_sizes_t = torch.tensor(
                        candidate_sizes, device=device, dtype=torch.long
                    )
                    chosen_length = candidate_sizes_t[chosen_idx]

                    old_latent_end_log_prob = group_end_log_probs[
                        torch.arange(B, device=device), chosen_idx
                    ]

                    inferred_latents = torch.stack(collected_latents[:-1], dim=1)  # (B, L, D)
                    latent_mask = torch.arange(L, device=device).unsqueeze(0) < chosen_length.unsqueeze(1)

                    # # Phase 3: Final Forward (reuse prefill cache, mask inactive groups)
                    prefix_len = (chosen_idx + 1) * (G + 1)
                    body_mask = (
                        torch.arange(body_len, device=device).unsqueeze(0) < prefix_len.unsqueeze(1)
                    ).to(attention_mask.dtype)

                    action_zero_embeds = torch.zeros(B, action_token_num, D, dtype=prev_latent_hidden.dtype, device=device)
                    action_mask_ones = torch.ones(B, action_token_num, dtype=attention_mask.dtype, device=device)

                    tail_mask = torch.cat([body_mask, action_mask_ones], dim=1)
                    full_attention = torch.cat([prefill_mask, tail_mask], dim=1)

                    cache_position_final = torch.arange(
                        prefill_len + L + max_groups, prefill_len + L + max_groups + action_token_num, device=device, dtype=torch.long
                    )

                    outputs = self.module.model(
                        inputs_embeds=action_zero_embeds,
                        pixel_values=None,
                        image_grid_thw=None,
                        attention_mask=full_attention,
                        cache_position=cache_position_final,
                        past_key_values=past_key_values,
                        return_dict=True, use_cache=False,
                        action_length=action_token_num,
                        attn_mode=self.config.attn_mode,
                    )
                    
                    hidden_states = outputs.last_hidden_state

                    value_hidden_states = None
                    if self.config.value_choice == 'action_mean':
                        value_hidden_states = hidden_states[:, -action_token_num:, :].mean(dim=1)
                    elif self.config.value_choice == 'latent_end':
                        value_hidden_states = for_value_hidden_states[
                            torch.arange(B, device=device), chosen_idx, :
                        ]
                    elif self.config.value_choice == 'prompt_last':
                        assert False, "prompt_last is not supported for Qwen-OFT"

                    assert value_hidden_states is not None, "value_hidden_states is None"
                    values = self.module.value_head(value_hidden_states).squeeze(-1)
                
                else:
                    assert False, "False"


                action_logits = self.module.lm_head(hidden_states[:, -action_token_num:, :])
                action_logits_last256 = action_logits[..., self.action_0_id:self.action_0_id + 256]
                if not torch.isfinite(action_logits_last256).all():
                    min_val = torch.finfo(action_logits_last256.dtype).min
                    print(
                        "[rob_rollout] Warning: non-finite values detected in action_logits_last256; "
                        "replacing nan/inf with dtype min."
                    )
                    action_logits_last256 = torch.nan_to_num(
                        action_logits_last256,
                        nan=min_val,
                        posinf=min_val,
                        neginf=min_val,
                    )
                if do_sample:
                    action_logits_last256 = action_logits_last256 / temperature
                    probs = torch.softmax(action_logits_last256, dim=-1)
                    sampled_indices = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1).squeeze(-1)
                    response = (sampled_indices + self.action_0_id).view(action_logits.shape[0], -1)
                    logprobs_tensor = F.log_softmax(action_logits_last256, dim=-1)
                else:
                    response = action_logits_last256.argmax(dim=-1) + self.action_0_id
                    logprobs_tensor = F.log_softmax(action_logits_last256, dim=-1)

                idxes = response.unsqueeze(-1) - self.action_0_id
                logprobs = torch.gather(logprobs_tensor, 2, idxes).squeeze(-1)
                logprobs = logprobs.sum(dim=1, keepdim=True).squeeze(-1)
                normalized_actions = self.action_tokenizer.decode_token_ids_to_actions(
                    response.cpu().numpy()
                ).reshape(-1, self.config.action_chunks_len, self.config.action_token_len)

                if normalized_actions.ndim == 1:
                    dim = len(normalized_actions)
                    if dim == 7 or dim == 14:
                        normalized_actions[6] = 0 if normalized_actions[6] < 0.5 else 1
                    if dim == 14:
                        normalized_actions[13] = 0 if normalized_actions[13] < 0.5 else 1
                else:
                    dim = normalized_actions.shape[-1]
                    if dim == 7 or dim == 14:
                        normalized_actions[..., 6] = (normalized_actions[..., 6] >= 0.5).astype(int)
                    if dim == 14:
                        normalized_actions[..., 13] = (normalized_actions[..., 13] >= 0.5).astype(int)

                actions = np.where(
                    self.action_mask,
                    0.5 * (normalized_actions + 1) * (self.action_max - self.action_min) + self.action_min,
                    normalized_actions,
                )

        if self.config.use_latent and latent_mask is None and inferred_latents is not None:
            latent_mask = torch.ones(B, L, device=device, dtype=torch.bool)

        batch = {
            'responses': response,
            'input_ids': idx,
            'attention_mask': full_attention if (self.config.use_latent and latent_end_num > 1 and latent_bind == 0) else attention_mask,
            "pixel_values": pixel_values,
            "action": actions,
            "old_log_probs": logprobs,
            "image_grid_thw": image_grid_thw,
            "old_values": values,
            "old_latents": inferred_latents,
            "old_latent_end_log_prob": old_latent_end_log_prob,
            "latent_mask": latent_mask,
            "chosen_length": chosen_length.to(device=device, dtype=torch.long)
        }
        
        return batch

    
    def _obs_to_input(self, obs, is_robotwin=False, robotwin_version="1.0"):
        """Convert observation to model input format"""
        if not is_robotwin:
            # Libero
            state = np.concatenate([
                obs["robot0_eef_pos"],
                quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"][:1]
            ])
            
            if self.config.num_images_in_input > 1:
                return {
                    "full_image": get_libero_image(obs, self.config.image_size),
                    "wrist_image": get_libero_wrist_image(obs, self.config.image_size),
                    "state": state
                }
            else:
                return {
                    "full_image": get_libero_image(obs, self.config.image_size),
                    "state": state
                }
        else:
            # Robotwin
            if robotwin_version == "1.0":
                state = obs['joint_action']
                state[6] /= 0.045
                state[13] /= 0.045
            else:  # 2.0
                state = obs['joint_action']['vector']
            
            if self.config.num_images_in_input == 3:
                return {
                    "full_image": obs['observation']['head_camera']['rgb'],
                    "left_wrist": obs['observation']['left_camera']['rgb'],
                    "right_wrist": obs['observation']['right_camera']['rgb'],
                    "state": state
                }
            else:
                return {
                    "full_image": obs['observation']['head_camera']['rgb'],
                    "state": state
                }
    
    def __del__(self):
        """Cleanup resources on deletion"""
        if hasattr(self, 'env_thread_pool'):
            self.env_thread_pool.shutdown(wait=False)