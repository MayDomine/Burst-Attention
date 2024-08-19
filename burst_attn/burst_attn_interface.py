import bmtrain as bmt
import torch
import torch.distributed as dist
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
from .comm import Ring, get_world_size, is_bmt_enable


def attn_forward(flash, q, k, v, m_i, lse_i, acc_o, scale, bias, causal=False):
    assert not causal or flash == "cuda", "Causal attention only supported for Flash v2"
    if flash == "triton":
        acc_o, m_i, lse_i = inter_flash_attn_triton(
            q, k, v, m_i, lse_i, acc_o, scale, bias
        )
    elif flash == "cuda":
        acc_o, lse_i = inter_flash_cuda_fwd(q, k, v, acc_o, lse_i, scale, causal=causal)
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
            bias,
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
        double_group=[None, None]
    ):
        m_i = None
        acc_o = None
        lse_i = None
        ctx.deterministic = deterministic
        assert (
            not causal or flash == "cuda"
        ), "Causal attention only supported for Flash v2"
        ctx.optimize_bwd_comm = optimize_bwd_comm
        if softmax_scale is None:
            ctx.softmax_scale = 1 / math.sqrt(q.shape[-1])
        else:
            ctx.softmax_scale = softmax_scale
        ctx.flash = None if flash not in ["cuda", "triton"] else flash
        ctx.causal = causal
        burst_comm = Ring(process_group, double_group)
        ctx.process_group = process_group
        ctx.double_group = double_group
        ctx.burst_comm = burst_comm
        ori_k, ori_v = k, v
        if causal:
            q1 = split2_gethalf(q, ctx.flash, 1)
        comm_bufs = [torch.zeros_like(t) for t in [k, v]]
        sp_count = burst_comm.world_size
        for r in range(1, sp_count + 1):
            if r != sp_count:
                burst_comm.double_ring_send_recv([k, v], comm_bufs, r)
                burst_comm.commit()
            if r == 1 or not causal:
                acc_o, _m_i, lse_i = attn_forward(
                    flash, q, k, v, m_i, lse_i, acc_o, ctx.softmax_scale, None, causal
                )
            elif r - 1 <= ctx.burst_comm.rank:
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
        acc_o = acc_o.to(dtype=q.dtype)
        lse_i = lse_i.squeeze(dim=-1).transpose(1, 2).contiguous()
        ctx.save_for_backward(q, ori_k, ori_v, lse_i, acc_o)
        return acc_o

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, lse_i, o_i = ctx.saved_tensors
        q = q.contiguous()
        lse_i = lse_i.contiguous()
        grad_output = grad_output.contiguous()
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        group, double_group = ctx.process_group, ctx.double_group
        dq_comm = Ring(group, double_group)
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

        burst_comm = ctx.burst_comm
        sp_count = ctx.burst_comm.world_size
        half_seqlen = q.shape[1] // 2 if ctx.flash else q.shape[2] // 2
        dqkv_buf = [torch.empty_like(t) for t in [dq, dk, dv]]
        if ctx.causal:
            k0 = split2_gethalf(k, ctx.flash).contiguous()
            v0 = split2_gethalf(v, ctx.flash).contiguous()
            dq_buf1 = split2_gethalf(dqkv_buf[0], ctx.flash).contiguous()
        read_comm_buf = [torch.empty_like(t) for t in [delta, grad_output, q, lse_i]]
        write_comm_buf = [torch.empty_like(dq)]
        for r in range(1, sp_count + 1):
            split_q = r - 1 <= ctx.burst_comm.rank
            if r != sp_count:
                burst_comm.double_ring_send_recv(
                    [delta, grad_output, q, lse_i], read_comm_buf, r
                )
                burst_comm.commit()
            if r != 1:
                dq_comm.double_ring_send_recv([dq], write_comm_buf, r)
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

        dq_comm.double_ring_send_recv([dq], write_comm_buf, r )
        dq_comm.commit()
        dq_comm.wait()
        dq = record_stream(*write_comm_buf)[0]

        return dq, dk, dv, None, None, None, None, None, None, None
