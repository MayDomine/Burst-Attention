import bmtrain as bmt
import mindspore as ms
import mindspore.ops as ops
from .comm import ring_bmt
import math
from .flash_attn import _flash_attn_forward,_flash_attn_backward
from bmtrain.distributed import  reduce_scatter

def ring(*tensor_list):
    res = []
    for t in tensor_list:
        res.append(ring_bmt(t))
    return res


def inter_normal_attn(q, k, v, m_i, acc_o, softmax_scale=1.0):
    m_i = m_i.to(q.dtype) if m_i is not None else None
    qk = q @ k.transpose(-2, -1)*softmax_scale
   
    m_ij = ops.max(qk, dim=-1, keepdim=True)[0]
    if m_i is not None:
        m_ij = ops.maximum(m_ij, m_i)
    p = ops.exp(qk - m_ij)
    l_ij = ops.sum(p, dim=-1, keepdim=True)
    if acc_o is not None:
        acc_o_scale = ops.exp(m_i-m_ij)
        pv = (p @ v).to(dtype=ms.float32)
        acc_o = pv + acc_o_scale * acc_o
    else:
        acc_o = (p @ v).to(dtype=ms.float32)
    return acc_o,m_ij,l_ij

def inter_normal_attn_backward(do, q, k, v, delta, lse, d_q, d_k, d_v, softmax_scale):
    qk = q @ k.transpose(-2, -1) * softmax_scale
    p = ops.exp(qk - lse) 
    d_v += p.transpose(-2, -1) @ do
    d_p = do @ v.transpose(-2, -1)
    softmax_scale = softmax_scale
    d_s = p * (d_p - delta) * softmax_scale
    d_q += d_s @ k
    d_k += d_s.transpose(-2, -1) @ q

def inter_flash_attn(q, k, v, m_i, lse_i, acc_o, softmax_scale=1.0):
    q = q.transpose(1,2)
    k = k.transpose(1,2)
    v = v.transpose(1,2)
    m_i = m_i.squeeze(-1) if m_i is not None else None
    if m_i is None:
        b,s,n,d = q.shape
        m_i = (-ops.ones((b,n,s),dtype=ms.float32) * ms.numpy.inf)
    if lse_i is None:
        b,s,n,d = q.shape
        lse_i = (-ops.ones((b,n,s),dtype=ms.float32) * ms.numpy.inf)
    if acc_o is None:
        acc_o = ops.zeros(list(q.shape),dtype=ms.float32)
    acc_o,lse_i,m_ij,softamx_scale = _flash_attn_forward(q,k,v,m_i,lse_i,acc_o.to(dtype=ms.float32),causal=False,softmax_scale=softmax_scale)
    
    m_ij = m_ij.unsqueeze(-1)
    lse_i = lse_i.unsqueeze(-1)
    return acc_o,m_ij,lse_i

def inter_flash_attn_backward(do,q,k,v,delta,lse,dq,dk,dv,softmax_scale):
    lse =  lse.squeeze(-1)
    delta = delta.squeeze(-1)
    do = do.transpose(1,2)
    q = q.transpose(1,2) #B N S D -> B S N H
    k = k.transpose(1,2) 
    v = v.transpose(1,2)
    dq_ = ops.empty_like(q)
    dk_ = ops.empty_like(k)
    dv_ = ops.empty_like(v)
    _flash_attn_backward(do, q, k, v, delta, lse, dq_, dk_, dv_, softmax_scale=softmax_scale)
    dq += dq_.transpose(1,2)
    dk += dk_.transpose(1,2)
    dv += dv_.transpose(1,2)


class OpBurstAttn(ops.autograd.Function):
    """
    q, k, v: [B, N, S, H] (batch_size, num_heads, sub_seqlen, head_dim)
    bias: [N, S, S] (num_heads, sub_seqlen, sub_seqlen)
    """
    @staticmethod
    def forward(ctx, query, key ,value, softmax_scale=None, flash=False):
        q = query
        k = key
        v = value
        if softmax_scale is None:
            ctx.softmax_scale = 1/math.sqrt(q.shape[-1])
        else:
            ctx.softmax_scale = softmax_scale
        m_i = None
        acc_o = None
        ctx.flash=flash
        lse_i = None
        if ctx.flash:
            forward_func = inter_flash_attn
        else:
            forward_func = inter_normal_attn
        k_buffer,v_buffer = ring(k,v)
        if ctx.flash:
            acc_o,m_i,lse_i = forward_func(q, k, v, m_i, lse_i, acc_o, ctx.softmax_scale)
        else:
            acc_o,m_i,l_ij = forward_func(q, k, v, m_i, acc_o, ctx.softmax_scale)
            lse_i = ops.log(l_ij) + m_i
        for j in range(bmt.world_size()-1):
            k,v = k_buffer, v_buffer
            k_buffer, v_buffer = ring(k, v)
            if not ctx.flash:
                with ops.no_grad():
                    acc_o,m_ij,l_ij = forward_func(q, k, v, m_i, acc_o, ctx.softmax_scale)
                    m_i = m_ij
                    l_i_new = ops.exp(lse_i - m_ij) + l_ij
                    lse_i = ops.log(l_i_new) + m_ij
            else:
                acc_o,m_i,lse_i = forward_func(q, k, v, m_i, lse_i, acc_o, ctx.softmax_scale)
        o_scale = ops.exp(m_i - lse_i)
        if ctx.flash:
            acc_o = acc_o.transpose(1,2)
        acc_o = acc_o * o_scale
        acc_o = acc_o.to(dtype=ops.float16)
        ctx.save_for_backward(query, key, value, lse_i, acc_o)
        return acc_o

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, lse_i, o_i = ctx.saved_tensors
        batch_size, num_heads, seqlen, head_dim = q.shape
        d_q_whole = ops.zeros(batch_size, num_heads, seqlen*bmt.world_size(), head_dim, device=q.device)
        d_q = ops.zeros_like(q)
        d_k = ops.zeros_like(k)
        d_v = ops.zeros_like(v)
        delta = (o_i * grad_output).sum(-1, keepdim=True)
        if ctx.flash:
            backward_func = inter_flash_attn_backward
        else:
            backward_func = inter_normal_attn_backward
        start = bmt.rank() * seqlen
        end = (bmt.rank() + 1) * seqlen
         
        delta_buffer, grad_output_buffer,q_buffer,lse_buffer = ring(delta, grad_output,q,lse_i)
        backward_func(grad_output, q, k, v, delta, lse_i, d_q_whole[:, :, start:end, :], d_k, d_v, ctx.softmax_scale)
        for j in range(bmt.world_size()-1):
            ops.cuda.current_stream().wait_stream(bmt.config["sp_stream"])
            delta, grad_output,q,lse_i = delta_buffer, grad_output_buffer,q_buffer,lse_buffer
            delta_buffer, grad_output_buffer,q_buffer,lse_buffer = ring(delta, grad_output,q,lse_i)
            start = (start - seqlen) % (seqlen * bmt.world_size())
            end = (end - seqlen) % (seqlen * bmt.world_size())
            if end < start:
                backward_func(grad_output, q, k, v, delta, lse_i, d_q_whole[:, :, start:, :], d_k, d_v, ctx.softmax_scale)
            else:
                backward_func(grad_output, q, k, v, delta, lse_i, d_q_whole[:, :, start:end, :], d_k, d_v, ctx.softmax_scale)
        # d_q_whole = rearrange(d_q_whole, "b n s h -> s n b h")
        d_q_whole = d_q_whole.tranpose(0,2)
        d_q = reduce_scatter(d_q_whole, "sum",bmt.config["comm"])
        # d_q = rearrange(d_q, "s n b h -> b n s h")
        d_q = d_q.transpose(0, 2)
        return d_q, d_k, d_v, None, None, None
    
