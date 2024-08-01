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


class OpBurstAttn(torch.autograd.Function):
    """
    for Normal Attention:
        q, k, v: [B, N, S, H] (batch_size, num_heads, sub_seqlen, head_dim)
    for Flash:
        q, k, v: [B, S, N, H] (batch_size, num_heads, sub_seqlen, head_dim)

    """

    @staticmethod
    def forward(
        ctx, q, k, v, softmax_scale=None, flash="cuda", optimize_bwd_comm=False, process_group=None
    ):
        m_i = None
        acc_o = None
        lse_i = None
        ctx.optimize_bwd_comm = optimize_bwd_comm
        if softmax_scale is None:
            ctx.softmax_scale = 1 / math.sqrt(q.shape[-1])
        else:
            ctx.softmax_scale = softmax_scale
        ctx.flash = None if flash not in ["cuda", "triton"] else flash
        if ctx.flash:
            forward_func = (
                inter_flash_attn_triton
                if ctx.flash == "triton"
                else inter_flash_cuda_fwd
            )
        else:
            forward_func = inter_normal_attn
        sp_count = get_world_size(process_group)

        burst_comm = Ring(process_group)
        ctx.burst_comm =burst_comm 

        for r in range(1, sp_count + 1):
            bufs = burst_comm.ring_send_recv(k, v)
            burst_comm.commit()
            if ctx.flash:
                if ctx.flash == "triton":
                    acc_o, m_i, lse_i = forward_func(
                        q, k, v, m_i, lse_i, acc_o, ctx.softmax_scale, None
                    )
                else:
                    acc_o, lse_i = forward_func(
                        q, k, v, acc_o, lse_i, ctx.softmax_scale
                    )
            else:
                acc_o, m_i, lse_i = forward_func(
                    q, k, v, m_i, lse_i, acc_o, ctx.softmax_scale, None
                )
            k, v = record_stream(*bufs)
            burst_comm.wait()

        if ctx.flash == "triton":
            acc_o = triton_scale_out(acc_o, m_i, lse_i)
        elif not ctx.flash:
            o_scale = torch.exp(m_i - lse_i)
            acc_o = acc_o * o_scale
        acc_o = acc_o.to(dtype=q.dtype)
        lse_i = lse_i.squeeze(dim=-1).transpose(1, 2).contiguous()
        ctx.save_for_backward(q, k, v, lse_i, acc_o)
        return acc_o

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, lse_i, o_i = ctx.saved_tensors
        d_q = torch.zeros_like(q)
        d_k = torch.zeros_like(k)
        d_v = torch.zeros_like(v)
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
        if ctx.flash:
            backward_func = (
                inter_flash_attn_backward_triton
                if ctx.flash == "triton"
                else inter_flash_cuda_bwd
            )
        else:
            backward_func = inter_normal_attn_backward

        burst_comm = ctx.burst_comm
        sp_count = ctx.burst_comm.world_size
        dq = torch.zeros_like(d_q)
        for r in range(1, sp_count + 1):
            # j = (i + sp_count - r) % sp_count

            if r != sp_count:
                bufs = burst_comm.ring_send_recv(delta, grad_output, q, lse_i)
            if r != 1:
                dq_buf = burst_comm.ring_send_recv(d_q)
            burst_comm.commit()
            backward_func(
                grad_output,
                q,
                k,
                v,
                delta,
                lse_i,
                dq,
                d_k,
                d_v,
                ctx.softmax_scale,
                None,
            )
            burst_comm.wait()
            if r != sp_count:
                delta, grad_output, q, lse_i = record_stream(*bufs)
                if is_bmt_enable():
                    torch.cuda.current_stream().wait_stream(bmt.config["sp_stream"])
            if r != 1:
                (d_q,) = record_stream(*dq_buf)
                d_q += dq
            else:
                d_q = dq.clone().detach()

        (d_q,) = burst_comm.ring_send_recv(d_q)
        burst_comm.commit()
        burst_comm.wait()

        return d_q, d_k, d_v, None, None, None, None

def attn_forward(flash, q, k, v, m_i, lse_i, acc_o, scale, bias, causal=False):
    assert not causal or flash == "cuda", "Causal attention only supported for Flash v2"
    if flash == "triton":
        acc_o, m_i, lse_i = inter_flash_attn_triton(q, k, v, m_i, lse_i, acc_o, scale, bias)
    elif flash == "cuda":
        acc_o, lse_i = inter_flash_cuda_fwd(q, k, v, acc_o, lse_i, scale, causal=causal)
        m_i = None
    else:
        acc_o, m_i, lse_i = inter_normal_attn(q, k, v, m_i, lse_i, acc_o, scale, bias)
    return acc_o, m_i, lse_i

def attn_backward(flash, grad_output, q, k, v, delta, lse, dq, dk, dv, scale, bias, causal=False):
    if flash == "cuda":
        inter_flash_cuda_bwd(grad_output, q, k, v, delta, lse, dq, dk, dv, scale, bias)
    elif flash == "triton":
        inter_flash_attn_backward_triton(grad_output, q, k, v, delta, lse, dq, dk, dv, scale, bias)
    else:
        inter_normal_attn_backward(grad_output, q, k, v, delta, lse, dq, dk, dv, scale, bias)

    # if ctx.flash:
    #     backward_func = (
    #         inter_flash_attn_backward_triton
    #         if ctx.flash == "triton"
    #         else inter_flash_cuda_bwd
    #     )
    # else:
    #     backward_func = inter_normal_attn_backward

def split2_gethalf(inp, flash, half_idx=0):
    if flash:
        if half_idx == 0:
            return inp[:, :inp.shape[1] // 2]
        else:
            return inp[:, inp.shape[1] // 2:]
    else:
        if half_idx == 0:
            return inp[:, :, :inp.shape[2] // 2]
        else:
            return inp[:, :, inp.shape[2] // 2:]

class OpBurstAttnCausal(torch.autograd.Function):
    """
    for Normal Attention:
        q, k, v: [B, N, S, H] (batch_size, num_heads, sub_seqlen, head_dim)
    for Flash:
        q, k, v: [B, S, N, H] (batch_size, num_heads, sub_seqlen, head_dim)

    """

    @staticmethod
    def forward(
        ctx, q, k, v, softmax_scale=None, flash="cuda", optimize_bwd_comm=False, process_group=None
    ):
        m_i = None
        acc_o = None
        lse_i = None
        ctx.optimize_bwd_comm = optimize_bwd_comm
        if softmax_scale is None:
            ctx.softmax_scale = 1 / math.sqrt(q.shape[-1])
        else:
            ctx.softmax_scale = softmax_scale
        ctx.flash = None if flash not in ["cuda", "triton"] else flash
        if ctx.flash:
            forward_func = (
                inter_flash_attn_triton
                if ctx.flash == "triton"
                else inter_flash_cuda_fwd
            )
        else:
            forward_func = inter_normal_attn
        sp_count = get_world_size(process_group)

        burst_comm = Ring(process_group)
        ctx.burst_comm =burst_comm 
        q1 = split2_gethalf(q, ctx.flash, 1)

        for r in range(1, sp_count + 1):
            bufs = burst_comm.ring_send_recv(k, v)
            burst_comm.commit()
            if r == 1:
                acc_o, _m_i, lse_i = attn_forward(flash, q, k, v, m_i, lse_i, acc_o, ctx.softmax_scale, None, True) 
            elif r <= ctx.burst_comm.rank:
                k0 = split2_gethalf(k, ctx.flash)
                v0 = split2_gethalf(v, ctx.flash)

                acc_o, _m_i, lse_i = attn_forward(flash, q, k0, v0, m_i, lse_i, acc_o, ctx.softmax_scale, None)
            else:
                acc_o, _m_i, lse_i = attn_forward(flash, q1, k, v, m_i, lse_i, acc_o, ctx.softmax_scale, None)



            if ctx.flash != "cuda":
                m_i = _m_i
            k, v = record_stream(*bufs)
            burst_comm.wait()

        if ctx.flash == "triton":
            acc_o = triton_scale_out(acc_o, m_i, lse_i)
        elif not ctx.flash:
            o_scale = torch.exp(m_i - lse_i)
            acc_o = acc_o * o_scale
        acc_o = acc_o.to(dtype=q.dtype)
        lse_i = lse_i.squeeze(dim=-1).transpose(1, 2).contiguous()
        ctx.save_for_backward(q, k, v, lse_i, acc_o)
        return acc_o

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, lse_i, o_i = ctx.saved_tensors
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
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
        k0 = split2_gethalf(k, ctx.flash)
        v0 = split2_gethalf(v, ctx.flash)
        half_seqlen = q.shape[1] if ctx.flash else q.shape[2] // 2
        dq_buf = torch.zeros_like(dq)
        dq_buf1 = split2_gethalf(dq_buf, ctx.flash)
        
        for r in range(1, sp_count + 1):
            q1 = split2_gethalf(q, ctx.flash, 1)
            d1 = split2_gethalf(delta, ctx.flash, 1)
            grad_output1 = split2_gethalf(grad_output, ctx.flash, 1)
            lse1 = split2_gethalf(lse_i, ctx.flash, 1)
            split_q = r <= ctx.burst_comm.rank
            if r != sp_count:
                bufs = burst_comm.ring_send_recv(delta, grad_output, q, lse_i)
            if r != 1:
                _dq_buf_comm = burst_comm.ring_send_recv(dq)
            burst_comm.commit()
            if r == 0:
                attn_backward(
                    ctx.flash,
                    grad_output,
                    q,
                    k,
                    v,
                    delta,
                    lse_i,
                    dq_buf1,
                    dk,
                    dv,
                    ctx.softmax_scale,
                    None,
                    True,
                )
            elif split_q:
                attn_backward(ctx.flash, grad_output1, q1, k, v, d1, lse1, dq_buf1, dk, dv, ctx.softmax_scale, None)
            else:
                attn_backward(ctx.flash, grad_output, q, k0, v0, delta, lse_i, dq, dk, dv, ctx.softmax_scale, None)

            burst_comm.wait()
            if r != sp_count:
                delta, grad_output, q, lse_i = record_stream(*bufs)
                if is_bmt_enable():
                    torch.cuda.current_stream().wait_stream(bmt.config["sp_stream"])
            if r != 1:
                (dq,) = record_stream(*_dq_buf_comm)
                if split_q and ctx.flash:
                    dq[:, half_seqlen:] += dq_buf1
                elif split_q:
                    dq[:, :, half_seqlen:] += dq_buf1
                else:
                    dq += dq_buf
            else:
                dq = dq.clone().detach()

        (dq,) = burst_comm.ring_send_recv(dq)
        burst_comm.commit()
        burst_comm.wait()

        return dq, dk, dv, None, None, None, None
