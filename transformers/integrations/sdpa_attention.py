import os
from typing import Optional

import torch

from ..utils import is_torch_npu_available, is_torch_xpu_available, logging
from ..utils.import_utils import is_torch_greater_or_equal


logger = logging.get_logger(__name__)
_SDPA_MASK_DEBUG_PRINTED = False


_is_torch_greater_or_equal_than_2_5 = is_torch_greater_or_equal("2.5", accept_dev=True)
_is_torch_greater_or_equal_than_2_8 = is_torch_greater_or_equal("2.8", accept_dev=True)
_is_torch_xpu_available = is_torch_xpu_available()
_is_torch_npu_available = is_torch_npu_available()


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def use_gqa_in_sdpa(attention_mask: Optional[torch.Tensor], key: torch.Tensor) -> bool:
    # GQA can only be used under the following conditions
    # 1.cuda
    #   - torch version >= 2.5
    #   - attention_mask is None (otherwise it will fall back to the math kernel)
    #   - key is not a torch.fx.Proxy (otherwise it will fail with a tracing error)
    # 2.xpu
    #   - torch version >= 2.8
    #   - key is not a torch.fx.Proxy (otherwise it will fail with a tracing error)
    # 3.npu
    #   - npu is not supported gqa currently
    if _is_torch_xpu_available:
        return _is_torch_greater_or_equal_than_2_8 and not isinstance(key, torch.fx.Proxy)
    if _is_torch_npu_available:
        return False
    return _is_torch_greater_or_equal_than_2_5 and attention_mask is None and not isinstance(key, torch.fx.Proxy)

# --------- attn_mode = "separate": latent full + action full (blocks independent) ----------- #

def build_mask_separate(q_length, kv_length, action_length, latent_length, device=None):
    query_global_positions = torch.arange(
        kv_length - q_length, kv_length, device=device, dtype=torch.long
    ).view(1, 1, q_length, 1)
    key_global_positions = torch.arange(kv_length, device=device, dtype=torch.long).view(
        1, 1, 1, kv_length
    )
    causal_mask = query_global_positions >= key_global_positions
    if action_length > 0:
        causal_mask[:, :, -action_length:, -action_length:] = True
    if latent_length > 0:
        causal_mask[:, :, -(latent_length + action_length):-action_length, -(latent_length + action_length):-action_length] = True
    return causal_mask


def change_mask_separate(attention_mask, action_length, latent_length):
    if action_length > 0:
        attention_mask[:, :, -action_length:, -action_length:] = True
    if latent_length > 0:
        attention_mask[:, :, -(latent_length + action_length):-action_length, -(latent_length + action_length):-action_length] = True
    return attention_mask

# --------- attn_mode = "full": entire sequence full attention ----------- #

def build_mask_full(q_length, kv_length, action_length, latent_length, device=None):
    causal_mask = torch.ones(q_length, kv_length, device=device, dtype=torch.bool).view(1, 1, q_length, kv_length)
    causal_mask[:, :, :, :] = True
    return causal_mask


def change_mask_full(mask, action_length, latent_length):
    B, headnum, L, _ = mask.shape
    last_row = mask[:, :, -1, :]
    last_row_expanded = last_row.unsqueeze(2)
    candidate_mask = last_row_expanded.expand(-1, -1, L, -1)
    row_all_false = (mask.sum(dim=-1) == 0)
    not_last_row = torch.arange(L, device=mask.device) != (L - 1)
    not_last_row = not_last_row.view(1, 1, L, 1)
    replace_condition = not_last_row & (~row_all_false.unsqueeze(-1))
    return torch.where(replace_condition, candidate_mask, mask)

# --------- attn_mode = "causal": latent causal + action full ----------- #

def build_mask_causal(q_length, kv_length, action_length, latent_length, device=None, **kwargs):
    query_global_positions = torch.arange(
        kv_length - q_length, kv_length, device=device, dtype=torch.long
    ).view(1, 1, q_length, 1)
    key_global_positions = torch.arange(kv_length, device=device, dtype=torch.long).view(
        1, 1, 1, kv_length
    )
    causal_mask = query_global_positions >= key_global_positions
    if action_length > 0:
        causal_mask[:, :, -action_length:, -action_length:] = True

    return causal_mask


def change_mask_causal(attention_mask, action_length, latent_length, **kwargs):
    if action_length > 0:
        attention_mask[:, :, -action_length:, -action_length:] = True

    return attention_mask

def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:

    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        logger.warning_once(
            "`sdpa` attention does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )
    sdpa_kwargs = {}
    if hasattr(module, "num_key_value_groups"):
        if not use_gqa_in_sdpa(attention_mask, key):
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
        else:
            sdpa_kwargs = {"enable_gqa": True}


    if 'action_length' in kwargs:
        action_length = kwargs.get('action_length')
        latent_length = kwargs.get('latent_length', 0)
        attn_mode = kwargs.get('attn_mode')

        if attn_mode == 'separate':
            if attention_mask is None:
                attention_mask = build_mask_separate(query.shape[-2], key.shape[-2], action_length, latent_length, device=query.device)
            else:
                attention_mask = change_mask_separate(attention_mask, action_length, latent_length)
        elif attn_mode == 'full':
            if attention_mask is None:
                attention_mask = build_mask_full(query.shape[-2], key.shape[-2], action_length, latent_length, device=query.device)
            else:
                attention_mask = change_mask_full(attention_mask, action_length, latent_length)
        elif attn_mode == 'causal':
            causal_extra = {}
            if kwargs.get('onepass_dynamic'):
                causal_extra['onepass_dynamic'] = True
            if kwargs.get('latent_end_num') is not None:
                causal_extra['latent_end_num'] = kwargs['latent_end_num']
            if kwargs.get('latent_mode') is not None:
                causal_extra['latent_mode'] = kwargs['latent_mode']
            if attention_mask is None:
                attention_mask = build_mask_causal(query.shape[-2], key.shape[-2], action_length, latent_length, device=query.device, **causal_extra)
            else:
                attention_mask = change_mask_causal(attention_mask, action_length, latent_length, **causal_extra)
        else:
            logger.warning_once(f"Invalid attention mode: {attn_mode}, using 'separate' as default")
            assert False, f"Invalid attention mode: {attn_mode}"

            
    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    global _SDPA_MASK_DEBUG_PRINTED
    if os.environ.get("QWEN_DEBUG_SDPA_MASK", "0") == "1" and attention_mask is not None and attention_mask.ndim == 4:
        if attention_mask.dtype == torch.bool:
            all_masked_rows = ~attention_mask.any(dim=-1)
        else:
            all_masked_rows = torch.isneginf(attention_mask).all(dim=-1)
        if all_masked_rows.any() and not _SDPA_MASK_DEBUG_PRINTED:
            n_rows = int(all_masked_rows.sum().item())
            logger.warning(
                "[QWEN_DEBUG_SDPA_MASK] found fully-masked query rows before SDPA: "
                f"rows={n_rows}, q={query.shape[-2]}, kv={key.shape[-2]}, "
                f"attn_mode={kwargs.get('attn_mode', 'unknown')}, "
                f"action_length={kwargs.get('action_length', 'n/a')}, "
                f"latent_length={kwargs.get('latent_length', 'n/a')}, "
                f"latent_end_num={kwargs.get('latent_end_num', 1)}, "
                f"onepass_dynamic={kwargs.get('onepass_dynamic', False)}"
            )
            _SDPA_MASK_DEBUG_PRINTED = True

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # Note that it is important to check first for the shape, otherwise compile will fail with `argument 'is_causal' must be bool, not SymBool`
    if is_causal is None:
        # The last condition is for encoder (decoder) models which specify this by passing their own `is_causal` flag
        # This is mainly due to those models having mixed implementations for encoder, decoder, and encoder-decoder attns
        is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    # When `is_causal = False` and the `attention_mask` is not of boolean type, the Ascend NPU's SDPA interface cannot utilize the FlashAttentionScore operator，
    # and falls back to small-operator concatenation. To invoke the FlashAttentionScore, the attention_mask must be converted to boolean type.
    # This adaptation ensures the `attention_mask` meets the requirement for using FlashAttentionScore.
    if _is_torch_npu_available:
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            # Convert to boolean type, making sdpa to force call FlashAttentionScore to improve performance.
            attention_mask = torch.logical_not(attention_mask.bool()).to(query.device)

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
        **sdpa_kwargs,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None
