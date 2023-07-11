import bmtrain as bmt
import torch
from einops import rearrange, repeat
from .comm import ring_bmt
import math
from .test_ring_attn import ring_attn
from .flash_attn import _flash_attn_forward,_flash_attn_backward
from bmtrain.distributed import send_activations, recv_activations, reduce_scatter, broadcast, all_gather
from .flash_origin import FlashAttnFunc

import subprocess

def inter_normal_attn(q, k, v, m_i, acc_o, softmax_scale=1.0):
    m_i = m_i.to(q.dtype) if m_i is not None else None
    qk = q @ k.transpose(-2, -1)*softmax_scale
   
    m_ij = torch.max(qk, dim=-1, keepdim=True)[0]
    if m_i is not None:
        m_ij = torch.maximum(m_ij, m_i)
    p = torch.exp(qk - m_ij)
    l_ij = torch.sum(p, dim=-1, keepdim=True)
    if acc_o is not None:
        acc_o_scale = torch.exp(m_i-m_ij)
        acc_o = p @ v + acc_o_scale * acc_o
    else:
        acc_o = p @ v
    return acc_o,m_ij,l_ij

def inter_normal_attn_backward(do, q, k, v, delta, lse, d_q, d_k, d_v, softmax_scale):
    qk = q @ k.transpose(-2, -1) * softmax_scale
    p = torch.exp(qk - lse) 
    d_v += p.transpose(-2, -1) @ do
    d_p = do @ v.transpose(-2, -1)
    softmax_scale = softmax_scale
    d_s = p * (d_p - delta) * softmax_scale
    d_q += d_s @ k
    d_k += d_s.transpose(-2, -1) @ q

def inter_flash_attn(q, k, v, m_i, acc_o, softmax_scale=1.0):
    q = q.transpose(1,2).contiguous()
    k = k.transpose(1,2).contiguous()
    v = v.transpose(1,2).contiguous()
    m_i = m_i.squeeze().contiguous() if m_i is not None else None
    if m_i is None:
        b,s,n,d = q.shape
        m_i = (-torch.ones((b,n,s),dtype=torch.float32,device="cuda") * torch.inf).contiguous()
    o,l_ij,m_ij,softamx_scale = _flash_attn_forward(q,k,v,m_i, causal=False,softmax_scale=softmax_scale)
    o = o.transpose(1,2).contiguous()
    if m_i is not None:
        m_ij = torch.maximum(m_i, m_ij)
    if acc_o is not None:
        acc_o_scale = torch.exp(m_i-m_ij).to(q.dtype)
        acc_o_scale = acc_o_scale.unsqueeze(-1).contiguous()
        acc_o = o + acc_o * acc_o_scale
    else:
        acc_o = o
    l_ij = torch.exp(l_ij - m_ij)
    m_ij = m_ij.unsqueeze(-1).contiguous()
    l_ij = l_ij.unsqueeze(-1).contiguous()
    return acc_o,m_ij,l_ij

def inter_flash_attn_backward(do,q,k,v,delta,lse,dq,dk,dv,softmax_scale):
    lse =  lse.squeeze(-1).contiguous()
    delta = delta.squeeze(-1).contiguous()
    do = do.transpose(1,2).contiguous()
    q = q.transpose(1,2).contiguous() #B N S D -> B S N H
    k = k.transpose(1,2).contiguous() 
    v = v.transpose(1,2).contiguous()
    dq_ = torch.empty_like(q)
    dk_ = torch.empty_like(k)
    dv_ = torch.empty_like(v)
    _flash_attn_backward(do, q, k, v, delta, lse, dq_, dk_, dv_, softmax_scale=softmax_scale)
    dq += dq_.transpose(1,2).contiguous()
    dk += dk_.transpose(1,2).contiguous()
    dv += dv_.transpose(1,2).contiguous()


class OpBurstAttn(torch.autograd.Function):
    """
    q, k, v: [B, N, S, H] (batch_size, num_heads, sub_seqlen, head_dim)
    bias: [N, S, S] (num_heads, sub_seqlen, sub_seqlen)
    """
    @staticmethod
    def forward(ctx, query, key ,value, softmax_scale=None, flash=False):
        
        q = query.contiguous()
        k = key.contiguous()
        v = value.contiguous()
        if softmax_scale is None:
            ctx.softmax_scale = 1/math.sqrt(q.shape[-1])
        else:
            ctx.softmax_scale = softmax_scale
        m_i = None
        acc_o = None
        ctx.flash=flash
        if ctx.flash:
            forward_func = inter_flash_attn
        else:
            forward_func = inter_normal_attn
        acc_o,m_i,l_ij = forward_func(q, k, v, m_i, acc_o, ctx.softmax_scale)
        lse_i = torch.log(l_ij) + m_i
        for j in range(bmt.world_size()-1):
            k = ring_bmt(k)
            v = ring_bmt(v)
            if not ctx.flash:
                with torch.no_grad():
                    acc_o,m_ij,l_ij = forward_func(q, k, v, m_i, acc_o, ctx.softmax_scale)
            else:
                acc_o,m_ij,l_ij = forward_func(q, k, v, m_i, acc_o, ctx.softmax_scale)
            m_i = m_ij
            l_i_new = torch.exp(lse_i - m_ij) + l_ij
            lse_i = torch.log(l_i_new) + m_ij
        o_scale = torch.exp(m_i - lse_i).to(q.dtype)
        acc_o = acc_o * o_scale
        ctx.save_for_backward(query, key, value, lse_i, acc_o)
        return acc_o

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, lse_i, o_i = ctx.saved_tensors
        batch_size, num_heads, seqlen, head_dim = q.shape
        d_q_whole = torch.zeros(batch_size, num_heads, seqlen*bmt.world_size(), head_dim, device=q.device)
        d_q = torch.zeros_like(q)
        d_k = torch.zeros_like(k)
        d_v = torch.zeros_like(v)
        delta = (o_i * grad_output).sum(-1, keepdim=True)
        if ctx.flash:
            backward_func = inter_flash_attn_backward
        else:
            backward_func = inter_normal_attn_backward
        start = bmt.rank() * seqlen
        end = (bmt.rank() + 1) * seqlen
        backward_func(grad_output, q, k, v, delta, lse_i, d_q_whole[:, :, start:end, :], d_k, d_v, ctx.softmax_scale)
        for j in range(bmt.world_size()-1):
            delta = ring_bmt(delta)
            q = ring_bmt(q)
            grad_output = ring_bmt(grad_output)
            lse_i = ring_bmt(lse_i)
            start = (start - seqlen) % (seqlen * bmt.world_size())
            end = (end - seqlen) % (seqlen * bmt.world_size())
            if end < start:
                backward_func(grad_output, q, k, v, delta, lse_i, d_q_whole[:, :, start:, :], d_k, d_v, ctx.softmax_scale)
            else:
                backward_func(grad_output, q, k, v, delta, lse_i, d_q_whole[:, :, start:end, :], d_k, d_v, ctx.softmax_scale)
        d_q_whole = rearrange(d_q_whole, "b n s h -> s n b h")
        d_q = reduce_scatter(d_q_whole, "sum",bmt.config["comm"])
        d_q = rearrange(d_q, "s n b h -> b n s h")
        return d_q, d_k, d_v, None, None, None
    
