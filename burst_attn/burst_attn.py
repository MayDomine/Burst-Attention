import bmtrain as bmt
import torch
from einops import rearrange, repeat
import math
from lao import _flash_attn_forward,_flash_attn_backward
from comm import ring_bmt
def record_stream(*tensorlist):
    for t in tensorlist:
        t.record_stream(torch.cuda.current_stream())
    return tensorlist

def async_ring(*tensor_list):

    res = []
    with torch.cuda.stream(bmt.config["sp_stream"]):
        for t in tensor_list:
            t.record_stream(bmt.config["sp_stream"])
            t = ring_bmt(t)
            res.append(t)
    return res

def inter_normal_attn(q, k, v, m_i, lse_i, acc_o, softmax_scale=1.0, mask_bias=None):
    m_i = m_i.to(q.dtype) if m_i is not None else None
    qk = q @ k.transpose(-2, -1) * softmax_scale
    if mask_bias is not None:
        qk = torch.masked_fill(
            qk,
            mask_bias == False,
            torch.scalar_tensor(float("-10000"), device=qk.device, dtype=qk.dtype),
        )
   
    m_ij = torch.max(qk, dim=-1, keepdim=True)[0]
    if m_i is not None:
        m_ij = torch.maximum(m_ij, m_i)
    p = torch.exp(qk - m_ij)
    if mask_bias is not None:
        p = torch.masked_fill(
            p,
            mask_bias == False,
            torch.scalar_tensor(float("0"), device=qk.device, dtype=qk.dtype),
        )
    l_ij = torch.sum(p, dim=-1, keepdim=True)
    if acc_o is not None:
        acc_o_scale = torch.exp(m_i-m_ij)
        pv = (p @ v).to(dtype=torch.float32)
        acc_o = pv + acc_o_scale * acc_o
    else:
        acc_o = (p @ v).to(dtype=torch.float32)

    if lse_i is None:
        lse_i = torch.log(l_ij+1e-5) + m_ij
    else:
        lse_i = torch.log(torch.exp(lse_i - m_ij) + l_ij+1e-5) + m_ij
    return acc_o,m_ij,lse_i

def inter_normal_attn_backward(do, q, k, v, delta, lse, d_q, d_k, d_v, softmax_scale, mask_bias):
    qk = q @ k.transpose(-2, -1) * softmax_scale
    if mask_bias is not None:
        qk = torch.masked_fill(
            qk,
            mask_bias == False,
            torch.scalar_tensor(float("-10000"), device=qk.device, dtype=qk.dtype),
        )
    p = torch.exp(qk - lse) 
    if mask_bias is not None:
        p = torch.masked_fill(
            p,
            mask_bias == False,
            torch.scalar_tensor(float("0"), device=qk.device, dtype=qk.dtype),
        )
    d_v += p.transpose(-2, -1) @ do
    d_p = do @ v.transpose(-2, -1)
    softmax_scale = softmax_scale
    d_s = p * (d_p - delta) * softmax_scale
    d_q += d_s @ k
    d_k += d_s.transpose(-2, -1) @ q
tensor = {}
def init_tensor(q):
    b,s,n,d = q.shape
    tensor['m_i'] = (-torch.ones((b,n,s),dtype=torch.float32,device="cuda") * torch.inf).contiguous()
    tensor['lse_i'] = (-torch.ones((b,n,s),dtype=torch.float32,device="cuda") * torch.inf).contiguous()
    tensor['acc_o'] = torch.zeros((b,s,n,d),dtype=torch.float32,device="cuda").contiguous()
    tensor['dq_'] = torch.empty_like(q)
    tensor['dk_'] = torch.empty_like(q)
    tensor['dv_'] = torch.empty_like(q)

def inter_flash_attn(q, k, v, m_i, lse_i, acc_o, softmax_scale = 1.0, mask_bias=None):
    if m_i is None:
        m_i = tensor['m_i']
        # m_i = (-torch.ones((b,n,s),dtype=torch.float32,device="cuda") * torch.inf).contiguous()
    if lse_i is None:
        lse_i = tensor['lse_i']
        # lse_i = (-torch.ones((b,n,s),dtype=torch.float32,device="cuda") * torch.inf).contiguous()
    if acc_o is None:
        # acc_o = torch.zeros(list(q.shape),dtype=torch.float32,device="cuda").contiguous()
        acc_o = tensor['acc_o']
    acc_o,lse_i,m_ij,softamx_scale = _flash_attn_forward(q,k,v,m_i,lse_i,acc_o.to(dtype=torch.float32),causal=False,bias=mask_bias,softmax_scale=softmax_scale)
    
    return acc_o,m_ij,lse_i

def inter_flash_attn_backward(do,q,k,v,delta,lse,dq,dk,dv,softmax_scale,mask_bias):
    dq_ = tensor['dq_']
    dk_ = tensor['dk_']
    dv_ = tensor['dv_']
    _flash_attn_backward(do, q, k, v, delta, lse, dq_, dk_, dv_, softmax_scale=softmax_scale,bias=mask_bias)
    dq += dq_
    dk += dk_
    dv += dv_

class OpBurstAttn(torch.autograd.Function):
    """
    q, k, v: [B, N, S, H] (batch_size, num_heads, sub_seqlen, head_dim)
    bias: [N, S, S] (num_heads, sub_seqlen, sub_seqlen)
    """
    @staticmethod
    def forward(ctx, q, k, v, softmax_scale=None, mask_bias=None, flash=False):
        init_tensor(q)
        m_i = None
        acc_o = None
        lse_i = None
        if softmax_scale is None:
            ctx.softmax_scale = 1/math.sqrt(q.shape[-1])
        else:
            ctx.softmax_scale = softmax_scale
        ctx.mask_bias=mask_bias
        ctx.flash=flash
        if ctx.flash:
            forward_func = inter_flash_attn
            seq_len = q.shape[-3]
        else:
            forward_func = inter_normal_attn
            seq_len = q.shape[-2]
        ctx.seq_len = seq_len
        i = bmt.rank()
        sp_count = bmt.world_size()
        # bmt.config['sp_stream'].wait_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()
        for r in range(1, sp_count+1):
            j = (i + sp_count - r) % sp_count
            bufs = async_ring(k, v)
            mb = mask_bias[0, :, :, j*seq_len:(j+1)*seq_len, :] if mask_bias is not None else None
            if ctx.flash:
                acc_o, m_i, lse_i = forward_func(q, k, v, m_i, lse_i, acc_o, ctx.softmax_scale, mb)
            else:
                acc_o, m_i, lse_i = forward_func(q, k, v, m_i, lse_i, acc_o, ctx.softmax_scale, mb)
            k,v = record_stream(*bufs)
            torch.cuda.current_stream().wait_stream(bmt.config['sp_stream'])
        o_scale = torch.exp(m_i - lse_i)
        if ctx.flash:
            acc_o = acc_o.transpose(1,2)
            o_scale = o_scale.unsqueeze(-1)
        acc_o = acc_o * o_scale
        acc_o = acc_o.to(dtype=torch.float16)
        ctx.save_for_backward(q, k, v, lse_i, acc_o)
        return acc_o

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, lse_i, o_i = ctx.saved_tensors
        d_q = torch.zeros_like(q)
        d_k = torch.zeros_like(k)
        d_v = torch.zeros_like(v)
        delta = (o_i * grad_output).sum(-1, keepdim=True)
        if ctx.flash:
            delta = delta.squeeze(-1)
            grad_output = grad_output.transpose(1,2).contiguous()
            backward_func = inter_flash_attn_backward
        else:
            backward_func = inter_normal_attn_backward
         
        seq_len = ctx.seq_len
        i = bmt.rank()
        # bmt.config['sp_stream'].wait_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()
        sp_count = bmt.world_size()
        for r in range(1, sp_count+1):
            j = (i + sp_count - r) % sp_count
            mb = ctx.mask_bias[1, :, :, j*seq_len:(j+1)*seq_len, :] if ctx.mask_bias is not None else None
            bufs = async_ring(delta, grad_output, q, d_q, lse_i)
            backward_func(grad_output, q, k, v, delta, lse_i, d_q, d_k, d_v, ctx.softmax_scale, mb)
            delta, grad_output, q, d_q, lse_i  = record_stream(*bufs)
            torch.cuda.current_stream().wait_stream(bmt.config['sp_stream'])
        return d_q, d_k, d_v, None, None, None
