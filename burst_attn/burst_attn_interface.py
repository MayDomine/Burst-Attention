import bmtrain as bmt
import torch
import math
from .burst_utils import (
    inter_normal_attn,
    inter_normal_attn_backward,
    inter_flash_attn_triton,
    inter_flash_attn_backward_triton,
    inter_flash_cuda_fwd,
    inter_flash_cuda_bwd,
)
from .burst_utils import triton_scale_out, record_stream
from .comm import Ring, get_world_size,  get_rank, replicate
from .log_helper import get_logger
from typing import Callable, Literal

_logger = get_logger(__file__, level="WARN")


def get_partition_id(double_group, r):
    inter_rank = get_rank(double_group[1])
    local_world_size = get_world_size(double_group[0])
    inter_size = get_world_size(double_group[1])
    intra_rank = get_rank(double_group[0])
    # double_round = (r - 1) // local_world_size
    double_round = 0
    round_r = (
        r - 1
        if double_group[0] is None
        else (inter_rank - ((r - 1) // local_world_size))
        % inter_size
        * local_world_size
        # + (r - 1) % local_world_size
        + (intra_rank - (r - 1) % local_world_size + local_world_size + double_round)
        % local_world_size
    )
    return round_r

def attn_forward(flash, q, k, v, m_i, lse_i, acc_o, scale, bias, causal=False, sliding_window=None, cu_seqlen=None):
    assert not causal or flash == "cuda", "Causal attention only supported for Flash v2"
    if flash == "triton":
        acc_o, m_i, lse_i = inter_flash_attn_triton(
            q, k, v, m_i, lse_i, acc_o, scale, bias
        )
    elif flash == "cuda":
        acc_o, lse_i = inter_flash_cuda_fwd(q, k, v, acc_o, lse_i, scale, causal=causal, sliding_window=sliding_window, cu_seqlen=cu_seqlen)
        m_i = None
    else:
        acc_o, m_i, lse_i = inter_normal_attn(q, k, v, m_i, lse_i, acc_o, scale, bias)
    return acc_o, m_i, lse_i


def attn_backward(
    flash,
    grad_output,
    q,
    k,
    v,
    delta,
    lse,
    dq,
    dk,
    dv,
    scale,
    bias,
    causal=False,
    cuda_args={},
):
    if flash == "cuda":
        return inter_flash_cuda_bwd(
            grad_output,
            q,
            k,
            v,
            delta,
            lse,
            dq,
            dk,
            dv,
            scale,
            causal,
            **cuda_args,
        )
    elif flash == "triton":
        return inter_flash_attn_backward_triton(
            grad_output, q, k, v, delta, lse, dq, dk, dv, scale, bias
        )
    else:
        return inter_normal_attn_backward(
            grad_output, q, k, v, delta, lse, dq, dk, dv, scale, bias
        )


def split2_gethalf(inp, first_dim, half_idx=0):
    if first_dim:
        if half_idx == 0:
            return inp[:, : inp.shape[1] // 2]
        else:
            return inp[:, inp.shape[1] // 2 :]
    else:
        if half_idx == 0:
            return inp[:, :, : inp.shape[2] // 2]
        else:
            return inp[:, :, inp.shape[2] // 2 :]


def burst_attn_func_striped(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float = None,
    flash: str = "cuda",
    causal: bool = False,
    optimize_bwd_comm: bool = False,
    deterministic: bool = False,
    process_group=None,
    double_group=[None, None],
    return_lse=False,
    grad_ckpt: Literal["seq-wise", "full", "none"] = "none",
    layer_idx: int = -1,
    sliding_window=None,
    cu_seqlen=None,
):
    assert grad_ckpt in [ "full", "seq-wise","none"], "grad_ckpt should be one of ''full', 'none', zig way do not support 'seq-wise'"
    if grad_ckpt != "none":
        grad_ckpt = (grad_ckpt, layer_idx)
    else:
        grad_ckpt = ("none", -1)


    return OpBurstAttnStrip.apply(
        q,
        k,
        v,
        softmax_scale,
        flash,
        causal,
        optimize_bwd_comm,
        deterministic,
        process_group,
        double_group,
        return_lse,
        grad_ckpt,
        sliding_window,
        cu_seqlen,
    )


def burst_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float = None,
    flash: str = "cuda",
    causal: bool = False,
    optimize_bwd_comm: bool = False,
    deterministic: bool = False,
    process_group=None,
    double_group=[None, None],
    return_lse=False,
    grad_ckpt: Literal["seq-wise", "full", "none"] = "none",
    layer_idx: int = -1,
):
    assert grad_ckpt in [ "full", "none"], "grad_ckpt should be one of ''full', 'none', zig way do not support 'seq-wise'"
    if grad_ckpt != "none":
        grad_ckpt = (grad_ckpt, layer_idx)
    else:
        grad_ckpt = ("none", -1)

    if k.storage().size() != k.size():
        k = replicate(k)
    if v.storage().size() != v.size():
        v = replicate(v)

    return OpBurstAttn.apply(
        q,
        k,
        v,
        softmax_scale,
        flash,
        causal,
        optimize_bwd_comm,
        deterministic,
        process_group,
        double_group,
        return_lse,
        grad_ckpt,
    )

_global_context = {}
def burst_attn_func_sparse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask_mod: Callable,
    softmax_scale: float = None,
    causal: bool = False,
    process_group=None,
    double_group=[None, None],
    return_lse=False,
    grad_ckpt: Literal["seq-wise", "full", "none"] = "none",
    layer_idx: int = -1,
):
    assert grad_ckpt in [ "full", "none"], "grad_ckpt should be one of ''full', 'none', zig way do not support 'seq-wise'"
    if grad_ckpt != "none":
        grad_ckpt = (grad_ckpt, layer_idx)
    else:
        grad_ckpt = ("none", -1)

    if k.storage().size() != k.size():
        k = replicate(k)
    if v.storage().size() != v.size():
        v = replicate(v)

    return OpBurstAttnBlockSparse.apply(
        q,
        k,
        v,
        mask_mod,
        softmax_scale,
        causal,
        process_group,
        double_group,
        return_lse,
        grad_ckpt,
    )

class OpBurstAttnFlex(torch.autograd.Function):
    """
    for Normal Attention:
        q, k, v: [B, N, S, H] (batch_size, num_heads, sub_seqlen, head_dim)
    for Flash:
        q, k, v: [B, S, N, H] (batch_size, num_heads, sub_seqlen, head_dim)

    """

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        mask_mod,
        softmax_scale=None,
        causal=False,
        process_group=None,
        double_group=[None, None],
        return_lse=False,
        grad_ckpt=None,
    ):
        acc_o = None
        lse_i = None
        if isinstance(double_group[0], tuple):
            dq_group = (double_group[0][1], double_group[1][1])
            double_group = (double_group[0][0], double_group[1][0])
            ctx.dq_group = dq_group
        else:
            # dq share same group with other tensors in backward
            ctx.dq_group = None
        if softmax_scale is None:
            ctx.softmax_scale = 1 / math.sqrt(q.shape[-1])
        else:
            ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        burst_comm = Ring(process_group, double_group)
        ctx.process_group = process_group
        ctx.double_group = double_group
        ori_k, ori_v = replicate(k), replicate(v)
        comm_bufs = [torch.zeros_like(t) for t in [k, v]]
        sp_count = burst_comm.world_size
        record = []
        _ctx = None
        _ctxs = []

        if grad_ckpt is not None:
            assert isinstance(grad_ckpt, tuple) and len(grad_ckpt) == 2
            ckpt_method, layer_idx = grad_ckpt
            ctx.layer_idx = layer_idx
        else:
            ckpt_method = "none"
            ctx.layer_idx = -1
        if isinstance(double_group[0], tuple) or isinstance(double_group[0], list):
            dq_group = (double_group[0][1], double_group[1][1])
            double_group = (double_group[0][0], double_group[1][0])
            ctx.dq_group = dq_group
        else:
            ctx.dq_group = None
            
        if layer_idx in _global_context and ckpt_method != "none":
            output = _global_context.pop(layer_idx)
            if output is not None:
                if ckpt_method == "full":
                    acc_o, lse_i = output
                    ctx.save_for_backward(q, k, v, lse_i, replicate(acc_o))
                    if return_lse:
                        return acc_o, lse_i
                    else:
                        return acc_o
            else:
                raise ValueError("No checkpointed output found for layer_idx {}".format(layer_idx))


        from .sparse_utils import (
            inter_flex_fwd
        )
        for r in range(1, sp_count + 1):
            round_r = get_partition_id(double_group, r)
            causal_shift = round_r > burst_comm.rank
            record.append(round_r)
            if r != sp_count:
                burst_comm.double_ring_send_recv([k, v], comm_bufs, r)
                burst_comm.commit()
            if not causal_shift or not causal:
                acc_o, lse_i, _ctx = inter_flex_fwd(
                    q,
                    k,
                    v,
                    lse_i,
                    acc_o,
                    ctx.softmax_scale,
                    causal,
                    mask_mod=mask_mod,
                )
            
            elif causal_shift:
                acc_o,  lse_i, _ctx = inter_flex_fwd(
                    q[:, :, 1:],
                    k[:, :, :-1],
                    v[:, :, :-1],
                    lse_i,
                    acc_o,
                    ctx.softmax_scale,
                    causal,
                    mask_mod,
                )

            if r != sp_count:
                kv, comm_bufs = record_stream(*comm_bufs), [k, v]
                k, v = kv
                burst_comm.wait()
            _ctx.pop("saved_tensors")
            _ctxs.append(_ctx)
        ctx._context = _ctxs
        _logger.info(f"Record of rank {get_rank()}: {record}")
        acc_o = acc_o.to(dtype=q.dtype)
        lse_i = lse_i.squeeze(dim=-1).contiguous()
        if layer_idx not in _global_context and ckpt_method != "none":
            if ckpt_method == "full":
                _global_context[layer_idx] = (acc_o.detach(), lse_i)
            else:
                raise ValueError("Invalid checkpoint method: {}".format(ckpt_method))

        ctx.save_for_backward(q, ori_k, ori_v, lse_i, acc_o)
        ctx.mask_mod = mask_mod
        if return_lse:
            return acc_o, lse_i
        else:
            return acc_o

    @staticmethod
    def backward(ctx, *grad_output):
        from .sparse_utils import (
            inter_flex_bwd
        )
        inter_flex_bwd = torch.compile(inter_flex_bwd)
        if isinstance(grad_output, tuple) and len(grad_output) == 2:
            grad_output, _ = grad_output
        else:
            grad_output = grad_output[0]
        q, k, v, lse_i, o_i = ctx.saved_tensors
        mask_mod  = ctx.mask_mod
        q = q.contiguous()
        lse_i = lse_i.contiguous()
        grad_output = grad_output.contiguous()
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        group, double_group = ctx.process_group, ctx.double_group
        dq_comm = Ring(
            group, ctx.dq_group if ctx.dq_group is not None else double_group
        )
        burst_comm = Ring(group, double_group)

        sp_count = burst_comm.world_size
        dqkv_buf = [torch.empty_like(t) for t in [dq, dk, dv]]
        read_comm_buf = [torch.empty_like(t) for t in [o_i, grad_output, q, lse_i]]
        write_comm_buf = [torch.empty_like(dq)]
        record = []
        for r in range(1, sp_count + 1):
            round_r = get_partition_id(double_group, r)
            record.append(round_r)
            causal_shift = round_r <= burst_comm.rank and r != 1
            if r != sp_count:
                burst_comm.double_ring_send_recv(
                    [o_i, grad_output, q, lse_i], read_comm_buf, r
                )
                burst_comm.commit()
            if r != 1:
                dq_comm.double_ring_send_recv_q([dq], write_comm_buf, r)
                dq_comm.commit()
            if not causal_shift or not ctx.causal:
                inter_flex_bwd(
                    ctx._context[r - 1],
                    grad_output,
                    q,
                    k,
                    v,
                    o_i,
                    lse_i,
                    mask_mod,
                    dqkv_buf[0],
                    dqkv_buf[1],
                    dqkv_buf[2],
                )
            elif causal_shift:
                lse = lse_i[:, :, 1:].contiguous()
                g = grad_output[:, :, 1:]
                inter_flex_bwd(
                    ctx._context[r - 1],
                    g,
                    q[:, :, 1:],
                    k[:, :, :-1],
                    v[:, :, :-1],
                    o_i[:, :, 1:],
                    lse,
                    mask_mod,
                    dqkv_buf[0][:, :, 1:],
                    dqkv_buf[1][:, :, 1:],
                    dqkv_buf[2][:, :, 1:],
                )

            if r != sp_count:
                recv, read_comm_buf = (
                    record_stream(*read_comm_buf),
                    [o_i, grad_output, q, lse_i],
                )
                delta, grad_output, q, lse_i = recv
            burst_comm.wait()
            if r != 1:
                dq_comm.wait()
                recv, write_comm_buf = record_stream(*write_comm_buf), [dq]
                dq = recv[0]
            if not causal_shift or not ctx.causal:
                dq += dqkv_buf[0]
                dk += dqkv_buf[1]
                dv += dqkv_buf[2]
            elif causal_shift:
                dq[:, 1:] += dqkv_buf[0][:, 1:]
                dk[:, :-1] += dqkv_buf[1][:, 1:]
                dv[:, :-1] += dqkv_buf[2][:, 1:]
        record.append(get_partition_id(double_group, sp_count + 1))
        _logger.info(f"Backward Record of rank {get_rank()}: {record}")
        dq_comm.double_ring_send_recv_q([dq], write_comm_buf, sp_count + 1)
        dq_comm.commit()
        dq_comm.wait()
        dq = record_stream(*write_comm_buf)[0]

        return dq, dk, dv, None, None, None, None, None, None, None

class OpBurstAttn(torch.autograd.Function):
    """
    for Normal Attention:
        q, k, v: [B, N, S, H] (batch_size, num_heads, sub_seqlen, head_dim)
    for Flash:
        q, k, v: [B, S, N, H] (batch_size, num_heads, sub_seqlen, head_dim)

    """

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        softmax_scale=None,
        flash="cuda",
        causal=False,
        optimize_bwd_comm=False,
        deterministic=False,
        process_group=None,
        double_group=[None, None],
        return_lse=False,
        grad_ckpt=None,
    ):
        m_i = None
        acc_o = None
        lse_i = None
        ctx.deterministic = deterministic
        if isinstance(double_group[0], tuple) or isinstance(double_group[0], list):
            dq_group = (double_group[0][1], double_group[1][1])
            double_group = (double_group[0][0], double_group[1][0])
            ctx.dq_group = dq_group
        else:
            # dq share same group with other tensors in backward
            ctx.dq_group = None

        if isinstance(process_group, list) or isinstance(process_group, tuple):
            ctx.global_dq_group = process_group[1]
            process_group = process_group[0]
        else:
            ctx.global_dq_group = process_group

        assert (
            not causal or flash == "cuda"
        ), "Causal attention only supported for Flash v2"
        ctx.optimize_bwd_comm = optimize_bwd_comm
        if softmax_scale is None:
            ctx.softmax_scale = 1 / math.sqrt(q.shape[-1])
        else:
            ctx.softmax_scale = softmax_scale
        ctx.double_group = double_group
        ctx.process_group = process_group
        ctx.flash = None if flash not in ["cuda", "triton"] else flash
        ctx.causal = causal
        burst_comm = Ring(process_group, double_group)

        if grad_ckpt is not None:
            assert isinstance(grad_ckpt, tuple) and len(grad_ckpt) == 2
            ckpt_method, layer_idx = grad_ckpt
            ctx.layer_idx = layer_idx
        else:
            ckpt_method = "none"
            ctx.layer_idx = -1
            
        if layer_idx in _global_context and ckpt_method != "none":
            output = _global_context.pop(layer_idx)
            if output is not None:
                if ckpt_method == "full":
                    acc_o, lse_i = output
                    ctx.save_for_backward(q, k, v, lse_i, replicate(acc_o))
                    if return_lse:
                        return acc_o, lse_i
                    else:
                        return acc_o
            else:
                raise ValueError("No checkpointed output found for layer_idx {}".format(layer_idx))

        ori_k, ori_v = replicate(k), replicate(v)
        if causal:
            q1 = split2_gethalf(q, ctx.flash, 1)
        comm_bufs = [torch.zeros_like(t) for t in [k, v]]
        sp_count = burst_comm.world_size
        record = []
        for r in range(1, sp_count + 1):
            round_r = get_partition_id(double_group, r)
            split_kv = round_r <= burst_comm.rank
            record.append(round_r)
            if r != sp_count:
                burst_comm.double_ring_send_recv([k, v], comm_bufs, r)
                burst_comm.commit()
            if r == 1 or not causal:
                acc_o, _m_i, lse_i = attn_forward(
                    flash, q, k, v, m_i, lse_i, acc_o, ctx.softmax_scale, None, causal
                )
            elif split_kv:
                k0 = split2_gethalf(k, ctx.flash)
                v0 = split2_gethalf(v, ctx.flash)

                acc_o, _m_i, lse_i = attn_forward(
                    flash, q, k0, v0, m_i, lse_i, acc_o, ctx.softmax_scale, None
                )
            else:
                acc_o, _m_i, lse_i = attn_forward(
                    flash, q1, k, v, m_i, lse_i, acc_o, ctx.softmax_scale, None
                )

            if ctx.flash != "cuda":
                m_i = _m_i
            if r != sp_count:
                kv, comm_bufs = record_stream(*comm_bufs), [k, v]
                k, v = kv
                burst_comm.wait()

        if ctx.flash == "triton":
            acc_o = triton_scale_out(acc_o, m_i, lse_i)
        elif not ctx.flash:
            o_scale = torch.exp(m_i - lse_i)
            acc_o = acc_o * o_scale
        _logger.info(f"Record of rank {get_rank()}: {record}")
        acc_o = acc_o.to(dtype=q.dtype)
        lse_i = lse_i.squeeze(dim=-1).transpose(1, 2).contiguous()
        if layer_idx not in _global_context and ckpt_method != "none":
            if ckpt_method == "full":
                _global_context[layer_idx] = (acc_o.detach(), lse_i)
            else:
                raise ValueError("Invalid checkpoint method: {}".format(ckpt_method))

        ctx.save_for_backward(q, ori_k, ori_v, lse_i, replicate(acc_o))
        if return_lse:
            return acc_o, lse_i
        else:
            return acc_o

    @staticmethod
    def backward(ctx, *grad_output):
        if isinstance(grad_output, tuple) and len(grad_output) == 2:
            grad_output, _ = grad_output
        else:
            grad_output = grad_output[0]
        q, k, v, lse_i, o_i = ctx.saved_tensors
        q = q.contiguous()
        lse_i = lse_i.contiguous()
        grad_output = grad_output.contiguous()
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        group, double_group = ctx.process_group, ctx.double_group
        dq_comm = Ring(
            ctx.global_dq_group, ctx.dq_group if ctx.dq_group is not None else double_group
        )
        burst_comm = Ring(group, double_group)
        if not ctx.optimize_bwd_comm:
            delta = o_i.contiguous()
        else:
            delta = (
                (o_i * grad_output)
                .to(dtype=torch.float32)
                .sum(-1, keepdim=not ctx.flash)
                .transpose(1, 2)
                .contiguous()
            )

        sp_count = burst_comm.world_size
        half_seqlen = q.shape[1] // 2 if ctx.flash else q.shape[2] // 2
        dqkv_buf = [torch.empty_like(t) for t in [dq, dk, dv]]
        if ctx.causal:
            k0 = split2_gethalf(k, ctx.flash).contiguous()
            v0 = split2_gethalf(v, ctx.flash).contiguous()
            dq_buf1 = split2_gethalf(dqkv_buf[0], ctx.flash).contiguous()
        read_comm_buf = [torch.empty_like(t) for t in [delta, grad_output, q, lse_i]]
        write_comm_buf = [torch.empty_like(dq)]
        record = []
        for r in range(1, sp_count + 1):
            round_r = get_partition_id(double_group, r)
            record.append(round_r)
            split_q = round_r <= burst_comm.rank
            if r != sp_count:
                burst_comm.double_ring_send_recv(
                    [delta, grad_output, q, lse_i], read_comm_buf, r
                )
                burst_comm.commit()
            if r != 1:
                dq_comm.double_ring_send_recv_q([dq], write_comm_buf, r)
                dq_comm.commit()
            if r == 1 or not ctx.causal:
                attn_backward(
                    ctx.flash,
                    grad_output,
                    q,
                    k,
                    v,
                    delta,
                    lse_i,
                    dqkv_buf[0],
                    dqkv_buf[1],
                    dqkv_buf[2],
                    ctx.softmax_scale,
                    None,
                    ctx.causal,
                    cuda_args={
                        "deterministic": ctx.deterministic,
                    },
                )
            elif split_q:
                q1 = split2_gethalf(q, ctx.flash, 1).contiguous()
                d1 = split2_gethalf(delta, not ctx.optimize_bwd_comm, 1).contiguous()
                grad_output1 = split2_gethalf(grad_output, ctx.flash, 1).contiguous()
                lse1 = split2_gethalf(lse_i, False, 1).contiguous()
                dq_buf1 = split2_gethalf(dqkv_buf[0], ctx.flash).contiguous()
                attn_backward(
                    ctx.flash,
                    grad_output1,
                    q1,
                    k,
                    v,
                    d1,
                    lse1,
                    dq_buf1,
                    dqkv_buf[1],
                    dqkv_buf[2],
                    ctx.softmax_scale,
                    None,
                    False,
                    cuda_args={
                        "deterministic": ctx.deterministic,
                    },
                )

            else:
                dk0 = torch.empty_like(split2_gethalf(dqkv_buf[1], ctx.flash, 0))
                dv0 = torch.empty_like(split2_gethalf(dqkv_buf[2], ctx.flash, 0))
                attn_backward(
                    ctx.flash,
                    grad_output,
                    q,
                    k0,
                    v0,
                    delta,
                    lse_i,
                    dqkv_buf[0],
                    dk0,
                    dv0,
                    ctx.softmax_scale,
                    None,
                    False,
                    cuda_args={
                        "deterministic": ctx.deterministic,
                    },
                )
            if r != sp_count:
                recv, read_comm_buf = (
                    record_stream(*read_comm_buf),
                    [delta, grad_output, q, lse_i],
                )
                delta, grad_output, q, lse_i = recv
            burst_comm.wait()
            if r != 1:
                dq_comm.wait()
                recv, write_comm_buf = record_stream(*write_comm_buf), [dq]
                dq = recv[0]
            if r == 1 or not ctx.causal:
                dq += dqkv_buf[0]
                dk += dqkv_buf[1]
                dv += dqkv_buf[2]
            elif split_q:
                dq[:, half_seqlen:] += dq_buf1
                dk += dqkv_buf[1]
                dv += dqkv_buf[2]
            else:
                dq += dqkv_buf[0]
                dk[:, :half_seqlen] += dk0
                dv[:, :half_seqlen] += dv0
        record.append(get_partition_id(double_group, sp_count + 1))
        _logger.info(f"Backward Record of rank {get_rank()}: {record}")
        dq_comm.double_ring_send_recv_q([dq], write_comm_buf, sp_count + 1)
        dq_comm.commit()
        dq_comm.wait()
        dq = record_stream(*write_comm_buf)[0]

        return dq, dk, dv, None, None, None, None, None, None, None, None, None


class OpBurstAttnStrip(torch.autograd.Function):
    """
    for Normal Attention:
        q, k, v: [B, N, S, H] (batch_size, num_heads, sub_seqlen, head_dim)
    for Flash:
        q, k, v: [B, S, N, H] (batch_size, num_heads, sub_seqlen, head_dim)

    """

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        softmax_scale=None,
        flash="cuda",
        causal=False,
        optimize_bwd_comm=False,
        deterministic=False,
        process_group=None,
        double_group=[None, None],
        return_lse=False,
        grad_ckpt=None,
        sliding_window=None,
        cu_seqlen = None,
    ):
        m_i = None
        acc_o = None
        lse_i = None
        ctx.deterministic = deterministic
        ctx.optimize_bwd_comm = optimize_bwd_comm
        if softmax_scale is None:
            ctx.softmax_scale = 1 / math.sqrt(q.shape[-1])
        else:
            ctx.softmax_scale = softmax_scale
        ctx.flash = None if flash not in ["cuda", "triton"] else flash
        ctx.causal = causal
        ctx.process_group = process_group
        ctx.sliding_window = sliding_window
        ctx.cu_seqlen = cu_seqlen
        if isinstance(double_group[0], tuple) or isinstance(double_group[0], list):
            dq_group = (double_group[0][1], double_group[1][1])
            double_group = (double_group[0][0], double_group[1][0])
            ctx.dq_group = dq_group
        else:
            # dq share same group with other tensors in backward
            ctx.dq_group = None
        ctx.double_group = double_group
        burst_comm = Ring(process_group, double_group)

        if grad_ckpt is not None:
            assert isinstance(grad_ckpt, tuple) and len(grad_ckpt) == 2
            ckpt_method,  layer_idx = grad_ckpt
            ctx.layer_idx = layer_idx
        else:
            ckpt_method = "none"
            ctx.layer_idx = -1

        ckpt_enable = ckpt_method != "none"
        calculate_half = False
        if  ckpt_enable and layer_idx in _global_context :
            output = _global_context.pop(layer_idx)
            if ckpt_method == "seq-wise":
                acc_o_half, lse_i_half = output
                calculate_half = True
            elif ckpt_method == "full":
                acc_o, lse_i = output
                ctx.save_for_backward(q, k, v, lse_i, replicate(acc_o))
                if return_lse:
                    return acc_o, lse_i
                else:
                    return acc_o
        ori_k, ori_v = replicate(k), replicate(v)
        if calculate_half:
            ori_q = q
            q = q[:, : q.shape[1] // 2].clone().contiguous()
            k = k[:, : k.shape[1] // 2].clone().contiguous()
            v = v[:, : v.shape[1] // 2].clone().contiguous()
        else:
            ori_q = q
        comm_bufs = [torch.zeros_like(t) for t in [k, v]]
        sp_count = burst_comm.world_size
        record = []
        for r in range(1, sp_count + 1):
            round_r = get_partition_id(double_group, r)
            causal_shift = round_r > burst_comm.rank
            record.append(round_r)
            if r != sp_count:
                burst_comm.double_ring_send_recv([k, v], comm_bufs, r)
                burst_comm.commit()
            if not causal_shift or not causal:
                acc_o, _m_i, lse_i = attn_forward(
                    flash, q, k, v, m_i, lse_i, acc_o, ctx.softmax_scale, None, causal, sliding_window, cu_seqlen
                )
            elif causal_shift:
                acc_o, _m_i, lse_i = attn_forward(
                    flash,
                    q[:, 1:],
                    k[:, :-1],
                    v[:, :-1],
                    m_i,
                    lse_i,
                    acc_o,
                    ctx.softmax_scale,
                    None,
                    causal,
                    sliding_window,
                    cu_seqlen,
                )

            if ctx.flash != "cuda":
                m_i = _m_i
            if r != sp_count:
                kv, comm_bufs = record_stream(*comm_bufs), [k, v]
                k, v = kv
                burst_comm.wait()

        if ctx.flash == "triton":
            acc_o = triton_scale_out(acc_o, m_i, lse_i)
        elif not ctx.flash:
            o_scale = torch.exp(m_i - lse_i)
            acc_o = acc_o * o_scale
        _logger.info(f"Record of rank {get_rank()}: {record}")
        acc_o = acc_o.to(dtype=q.dtype)
        lse_i = lse_i.squeeze(dim=-1).transpose(1, 2).contiguous()
        if layer_idx not in _global_context and ckpt_enable:
            if ckpt_method == "seq-wise":
                _global_context[layer_idx] = (replicate(acc_o.detach().chunk(2, dim=1)[1]), replicate(lse_i.chunk(2, dim=2)[1]))
            elif ckpt_method == "full":
                _global_context[layer_idx] = (acc_o.detach(), lse_i)
        if calculate_half:
            acc_o = torch.cat([acc_o, acc_o_half], dim=1).detach().contiguous()
            lse_i = torch.cat([lse_i, lse_i_half], dim=2).detach().contiguous()
            _global_context.pop(ctx.layer_idx)
        ctx.save_for_backward(ori_q, ori_k, ori_v, lse_i, replicate(acc_o))
        if return_lse:
            return acc_o, lse_i
        else:
            return acc_o

    @staticmethod
    def backward(ctx, *grad_output):
        if isinstance(grad_output, tuple) and len(grad_output) == 2:
            grad_output, _ = grad_output
        else:
            grad_output = grad_output[0]
        q, k, v, lse_i, o_i = ctx.saved_tensors
        q = q.contiguous()
        lse_i = lse_i.contiguous()
        grad_output = grad_output.contiguous()
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        group, double_group = ctx.process_group, ctx.double_group
        dq_comm = Ring(
            group, ctx.dq_group if ctx.dq_group is not None else double_group
        )
        burst_comm = Ring(group, double_group)
        if not ctx.optimize_bwd_comm:
            delta = o_i.contiguous()
        else:
            delta = (
                (o_i * grad_output)
                .to(dtype=torch.float32)
                .sum(-1, keepdim=not ctx.flash)
                .transpose(1, 2)
                .contiguous()
            )

        sp_count = burst_comm.world_size
        dqkv_buf = [torch.empty_like(t) for t in [dq, dk, dv]]
        read_comm_buf = [torch.empty_like(t) for t in [delta, grad_output, q, lse_i]]
        write_comm_buf = [torch.empty_like(dq)]
        record = []
        for r in range(1, sp_count + 1):
            round_r = get_partition_id(double_group, r)
            record.append(round_r)
            causal_shift = round_r <= burst_comm.rank and r != 1
            if r != sp_count:
                burst_comm.double_ring_send_recv(
                    [delta, grad_output, q, lse_i], read_comm_buf, r
                )
                burst_comm.commit()
            if r != 1:
                dq_comm.double_ring_send_recv_q([dq], write_comm_buf, r)
                dq_comm.commit()
            if not causal_shift or not ctx.causal:
                attn_backward(
                    ctx.flash,
                    grad_output,
                    q,
                    k,
                    v,
                    delta,
                    lse_i,
                    dqkv_buf[0],
                    dqkv_buf[1],
                    dqkv_buf[2],
                    ctx.softmax_scale,
                    None,
                    ctx.causal,
                    cuda_args={
                        "deterministic": ctx.deterministic,
                        "sliding_window": ctx.sliding_window,
                        "cu_seqlen": ctx.cu_seqlen,
                    },
                )
            elif causal_shift:
                if ctx.optimize_bwd_comm:
                    d = torch.zeros_like(delta)
                    d[:,:, :-1].copy_(delta[:, :, 1:])
                else:
                    d = delta[:, 1:]
                lse = lse_i[:, :, 1:].contiguous()
                if ctx.flash:
                    g = grad_output[:, 1:]
                else:
                    g = grad_output[:, :, 1:]
                attn_backward(
                    ctx.flash,
                    g,
                    q[:, 1:],
                    k[:, :-1],
                    v[:, :-1],
                    d,
                    lse,
                    dqkv_buf[0][:, 1:],
                    dqkv_buf[1][:, 1:],
                    dqkv_buf[2][:, 1:],
                    ctx.softmax_scale,
                    None,
                    ctx.causal,
                    cuda_args={
                        "deterministic": ctx.deterministic,
                        "sliding_window": ctx.sliding_window,
                        "cu_seqlen": ctx.cu_seqlen,
                    },
                )

            if r != sp_count:
                recv, read_comm_buf = (
                    record_stream(*read_comm_buf),
                    [delta, grad_output, q, lse_i],
                )
                delta, grad_output, q, lse_i = recv
            burst_comm.wait()
            if r != 1:
                dq_comm.wait()
                recv, write_comm_buf = record_stream(*write_comm_buf), [dq]
                dq = recv[0]
            if not causal_shift or not ctx.causal:
                dq += dqkv_buf[0]
                dk += dqkv_buf[1]
                dv += dqkv_buf[2]
            elif causal_shift:
                dq[:, 1:] += dqkv_buf[0][:, 1:]
                dk[:, :-1] += dqkv_buf[1][:, 1:]
                dv[:, :-1] += dqkv_buf[2][:, 1:]
        record.append(get_partition_id(double_group, sp_count + 1))
        _logger.info(f"Backward Record of rank {get_rank()}: {record}")
        dq_comm.double_ring_send_recv_q([dq], write_comm_buf, sp_count + 1)
        dq_comm.commit()
        dq_comm.wait()
        dq = record_stream(*write_comm_buf)[0]

        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None
