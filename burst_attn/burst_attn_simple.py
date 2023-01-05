import bmtrain as bmt
import torch
from bmtrain.distributed import send_activations, recv_activations, reduce_scatter, broadcast, ring_send_recv, all_gather
import os
from einops import rearrange, repeat
import bmtrain.nccl as nccl
def ring_bmt(tensor):
    return ring_send_recv(tensor, bmt.rank(), bmt.config["comm"])

def inter_attn(q, k, v, m_i, acc_o, softmax_scale=1.0):
    qk = q @ k.transpose(-2, -1)*softmax_scale
    m_ij = torch.maximum(torch.max(qk, dim=-1, keepdim=True)[0], m_i)
    p = torch.exp(qk - m_ij)
    l_ij = torch.sum(p, dim=-1, keepdim=True)
    acc_o_scale = torch.exp(m_i-m_ij)
    acc_o = p @ v + acc_o_scale * acc_o
    return m_ij, l_ij, acc_o


class OpBurstAttn(torch.autograd.Function):
    """
    q, k, v: [B, N, S, H] (batch_size, num_heads, sub_seqlen, head_dim)
    bias: [N, S, S] (num_heads, sub_seqlen, sub_seqlen)
    """
    @staticmethod
    def forward(ctx, q, k ,v, mask, softmax_scale=1.0):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        ctx.softmax_scale = softmax_scale
        batch_size, num_heads, seqlen, head_dim = q.shape
        qk = q @ k.transpose(-2, -1)*softmax_scale
        m_i = torch.max(qk, dim=-1, keepdim=True)[0]
        p = torch.exp(qk - m_i)
        l_ij = torch.sum(p, dim=-1, keepdim=True)
        lse_i = torch.log(l_ij) + m_i
        acc_o = p @ v
        for j in range(bmt.world_size()-1):
            k = ring_bmt(k)
            v = ring_bmt(v)
            m_ij, l_ij, acc_o = inter_attn(q, k, v, m_i, acc_o, softmax_scale=1.0)
            m_i = m_ij
            l_i_new = torch.exp(lse_i - m_ij) + l_ij
            lse_i = torch.log(l_i_new) + m_ij
        o_scale = torch.exp(m_i - lse_i)
        acc_o = acc_o * o_scale
        ctx.save_for_backward(q, k, v, m_i, lse_i, acc_o)
        return acc_o

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, m_i, lse_i, o_i = ctx.saved_tensors
        batch_size, num_heads, seqlen, head_dim = q.shape
        delta = (o_i * grad_output).sum(-1)
        qk = q @ k.transpose(-2, -1)
        p = torch.exp(qk - m_i) / lse_i
        d_v = p.transpose(-2, -1) @ grad_output
        d_p = grad_output @ v.transpose(-2, -1)
        d_s = p * (d_p - delta)
        d_q_whole = torch.zeros(batch_size, num_heads, seqlen*bmt.world_size(), head_dim, device=q.device)
        start = bmt.rank() * seqlen
        end = (bmt.rank() + 1) * seqlen
        d_q_whole[:, :, start:end, :] = d_s @ k
        d_k = d_s.T @ q
        softmax_scale = ctx.softmax_scale
        for j in range(bmt.world_size()-1):
            delta = ring_send_recv(delta)
            q = ring_send_recv(q)
            grad_output = ring_send_recv(grad_output)
            lse_i = ring_send_recv(lse_i)
            qk = q @ k.transpose(-2, -1)
            qk = qk * softmax_scale 
            p = torch.exp(qk) / lse_i
            d_v += p.transpose(-2, -1) @ grad_output
            d_p = grad_output @ v.transpose(-2, -1)
            d_s = p * (d_p - delta) * softmax_scale
            d_k += d_s.T @ q
            start = (start + seqlen) % (seqlen * bmt.world_size())
            end = (end + seqlen) % (seqlen * bmt.world_size())
            if end < start:
                d_q_whole[:, :, start:, :] += d_s @ k
            else:
                d_q_whole[:, :, start:end, :] = d_s @ k
        d_q_whole = rearrange(d_q_whole, "b n s h -> s b n h")
        d_q = reduce_scatter(d_q_whole, bmt.config["zero_comm"])
        d_q = rearrange(d_q, "s b n h -> b n s h")
        return d_q, d_k, d_v, None, None, None

def test():
    def ref_attn(q, k, v):
        s = q @ k.transpose(-2, -1)
        s = torch.softmax(s, dim=-1)
        p = s @ v
        return p
    batch = 2
    seqlen = 1024
    head_dim = 32
    num_heads = 16
    bmt.init_distributed()
    q_whole = torch.randn((batch, num_heads, seqlen, head_dim),device="cuda")
    k_whole = torch.randn((batch, num_heads, seqlen, head_dim),device="cuda")
    v_whole = torch.randn((batch, num_heads, seqlen, head_dim),device="cuda")
    q_whole = broadcast(q_whole, 0, bmt.config["comm"])
    k_whole = broadcast(k_whole, 0, bmt.config["comm"])
    v_whole = broadcast(v_whole, 0, bmt.config["comm"])
    sub_seq = seqlen // bmt.world_size()
    q = q_whole[:, :, bmt.rank()*sub_seq:(bmt.rank()+1)*sub_seq, :]
    k = k_whole[:, :, bmt.rank()*sub_seq:(bmt.rank()+1)*sub_seq, :]
    v = v_whole[:, :, bmt.rank()*sub_seq:(bmt.rank()+1)*sub_seq, :]
    res = OpBurstAttn.apply(q, k, v, None)
    res = rearrange(res, "b n s h -> s n b h")
    res = all_gather(res, bmt.config["comm"]).flatten(0,1)
    res_whole = rearrange(res, "s n b h -> b n s h")
    res2 = ref_attn(q_whole, k_whole, v_whole)
    print(torch.allclose(res_whole, res2, atol=1e-5))
    
if __name__ == "__main__":
    test()
        