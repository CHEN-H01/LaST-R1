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

from .base import BasePPOActor
from .dp_rob import RobDataParallelPPOActor
from .action_tokenizer import ActionTokenizer

# Lazy import to avoid loading flash_attn when only RobDataParallelPPOActor is used (avoids PyTorch ABI mismatch)
def __getattr__(name):
    if name == "DataParallelPPOActor":
        from .dp_actor import DataParallelPPOActor
        return DataParallelPPOActor
    if name == "DataParallelPRIME":
        from .dp_prime import DataParallelPRIME
        return DataParallelPRIME
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["BasePPOActor", "DataParallelPPOActor", "DataParallelPRIME", "RobDataParallelPPOActor", "ActionTokenizer"]
