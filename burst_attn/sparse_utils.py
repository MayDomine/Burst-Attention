import math
from einops import rearrange
from functools import lru_cache
from torch._higher_order_ops.flex_attention import flex_attention as flex_attention_hop
from torch._higher_order_ops.flex_attention import (
    create_fw_bw_graph,
    flex_attention_backward,
)

import torch

from torch.nn.attention.flex_attention import (
    create_block_mask,
    _identity,
)
from typing import Any, Dict, Tuple



def build_graph(tensor, score_mod, score_mod_other_buffers=()):
    example_vals = [torch.zeros((), dtype=tensor.dtype, requires_grad=True)] + [
        torch.zeros((), dtype=torch.int) for _ in range(4)
    ]
    fw_graph, joint_graph = create_fw_bw_graph(
        score_mod, example_vals, score_mod_other_buffers
    )
    return fw_graph, joint_graph


@torch.jit.script
def scale_out_lse(
    o,
    lse,
    o_i,
    lse_i,
):
    o_i = o_i.to(torch.float32)
    lse_i = lse_i.unsqueeze(dim=-1).contiguous()
    o = o - torch.sigmoid(lse_i - lse) * (o - o_i)
    lse = lse - torch.nn.functional.logsigmoid(lse - lse_i)
    return o, lse

def flex_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    fw_graph,
    joint_graph,
    block_mask,
    scale,
    kernel_options,
    score_mod_other_buffers=(),
    mask_mod_other_buffers=(),
):
    any_buffer_requires_grad = any(
        buffer.requires_grad
        for buffer in score_mod_other_buffers + mask_mod_other_buffers
    )
    assert (
        not any_buffer_requires_grad
    ), "Captured buffers that require grad are not yet supported."

    # Save necessary context for backward pass
    block_mask = block_mask.as_tuple()
    ctx = {
        "_fw_graph": fw_graph,
        "_joint_graph": joint_graph,
        "_mask_graph": block_mask[-1],
        "_KV_BLOCK_SIZE": block_mask[8],
        "_Q_BLOCK_SIZE": block_mask[9],
        "scale": scale,
        "kernel_options": kernel_options,
        "_score_mod_other_buffers_len": len(score_mod_other_buffers),
    }

    with torch._C._AutoDispatchBelowAutograd():
        out, logsumexp = flex_attention_hop(
            query,
            key,
            value,
            fw_graph,
            block_mask,
            scale,
            kernel_options,
            score_mod_other_buffers,
            mask_mod_other_buffers,
        )

    # Save tensors for backward pass
    ctx["saved_tensors"] = (
        query,
        key,
        value,
        out,
        logsumexp,
        *block_mask[:8],
        *score_mod_other_buffers,
        *mask_mod_other_buffers,
    )

    return out, logsumexp * math.log(2), ctx

def flex_backward(
    ctx: Dict[str, Any],
    grad_out: torch.Tensor,
    grad_logsumexp: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fw_args = ctx["saved_tensors"]
    (
        query,
        key,
        value,
        out,
        logsumexp,
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        q_num_blocks,
        q_indices,
        full_q_num_blocks,
        full_q_indices,
        *other_buffers,
    ) = fw_args
    
    fw_graph = ctx["_fw_graph"]
    joint_graph = ctx["_joint_graph"]
    mask_graph = ctx["_mask_graph"]
    KV_BLOCK_SIZE = ctx["_KV_BLOCK_SIZE"]
    Q_BLOCK_SIZE = ctx["_Q_BLOCK_SIZE"]
    scale = ctx["scale"]
    kernel_options = ctx["kernel_options"]
    score_mod_other_buffers = tuple(
        other_buffers[: ctx["_score_mod_other_buffers_len"]]
    )
    mask_mod_other_buffers = tuple(
        other_buffers[ctx["_score_mod_other_buffers_len"] :]
    )
    
    # We have asserted that other_buffers do not require grad in the forward
    none_grads = [None] * 7
    
    grad_query, grad_key, grad_value = flex_attention_backward(
        query,
        key,
        value,
        out,
        logsumexp,
        grad_out,
        grad_logsumexp,
        fw_graph,
        joint_graph,
        (
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            q_num_blocks,
            q_indices,
            full_q_num_blocks,
            full_q_indices,
            KV_BLOCK_SIZE,
            Q_BLOCK_SIZE,
            mask_graph,
        ),
        scale,
        kernel_options,
        score_mod_other_buffers,
        mask_mod_other_buffers,
    )
    
    return grad_query, grad_key, grad_value, *none_grads

@lru_cache
def create_block_mask_cached(mask_mod, B, H, M, N, device="cuda"):
    block_mask = create_block_mask(mask_mod, B, H, M, N, device=device)
    return block_mask


def generate_block_mask(mask_mod, B, H, S, block_size):
    block_mask = create_block_mask_cached(mask_mod, B, H, S, S)


def reorder_input(input, block_size):
    # incase block_size = 2
    # [1, 2, 3, 4, 5, 6, 7, 8] -> [1, 3, 5, 7], [2, 4, 6, 8] -> [1, 3, 5, 7, 2, 4, 6, 8]
    # block_size should always set to group size of BurstAttention
    B, S, N, D = input.shape
    block_inp = input.reshape(B, S // block_size, block_size, N, D)
    block_inp = rearrange(
        block_inp.transpose(1, 2), "B b s n d -> B n (b s) d"
    ).contiguous()
    return block_inp


@torch.compiler.disable(recursive=False)
def inter_flex_fwd(
    q, k, v, lse, o, softmax_scale=1.0, causal=False, mask_mod=None, score_mod=None
):
    B, H, Sq, D = q.shape
    if score_mod is None:
        score_mod = _identity
    score_mod_other_buffers = ()
    mask_mod_other_buffers = ()
    # if causal:
    #     _mask_mod = mask_mod
    #     def causal_mask_mod(b, h, q_idx, kv_idx):
    #         causal_mask = q_idx >= kv_idx & _mask_mod(b, h, q_idx, kv_idx)
    #         return causal_mask
    #
    #     mask_mod = causal_mask_mod
    #
    fw_graph, joint_graph = build_graph(q, score_mod, score_mod_other_buffers)
    S = q.shape[2]
    # block_mask = create_block_mask_cached(mask_mod, 1, 1, S, S, device=q.device)
    scale = 1.0 / math.sqrt(D)
    kernel_options = {}

    o_i, lse_i, ctx = flex_forward(
        q,
        k,
        v,
        fw_graph,
        joint_graph,
        mask_mod,
        scale,
        kernel_options,
        score_mod_other_buffers,
        mask_mod_other_buffers,
    )
    if o is None:
        o = o_i.to(torch.float32)
        lse = lse_i.unsqueeze(dim=-1).contiguous()
    else:
        if q.shape[2] == k.shape[2] // 2:
            half_seqlen = o.shape[1] // 2

            o[:, :, half_seqlen:], lse[:, :, half_seqlen:] = scale_out_lse(
                o[:, :, half_seqlen:], lse[:, :, half_seqlen:], o_i, lse_i
            )
        elif lse.shape[2] == lse_i.shape[2] + 1:
            o[:, :, 1:], lse[:, :, 1:] = scale_out_lse(
                o[:, :, 1:], lse[:, :, 1:], o_i, lse_i
            )
        else:
            o, lse = scale_out_lse(o, lse, o_i, lse_i)
    return o, lse, ctx

@torch.compiler.disable(recursive=False)
def inter_flex_bwd(
    ctx,
    do,
    q,
    k,
    v,
    o,
    lse,
    mask_mod,
    dq,
    dk,
    dv,
):

    B, H, Sq, D = q.shape
    Sk = k.shape[2]
    score_mod_other_buffers = ()
    mask_mod_other_buffers = ()
    block_mask = mask_mod.as_tuple()
    ctx["saved_tensors"] = (
        q,
        k,
        v,
        o,
        lse / math.log(2),
        *block_mask[:8],
        *score_mod_other_buffers,
        *mask_mod_other_buffers,
    )
    grad_query, grad_key, grad_value, *_ = flex_backward(ctx, do, torch.zeros_like(lse))
    dq += grad_query
    dk += grad_key
    dv += grad_value




