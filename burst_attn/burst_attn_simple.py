import bmtrain as bmt
import torch
from bmtrain.distributed import send_activations, recv_activations, reduce_scatter, broadcast, all_gather
from einops import rearrange, repeat
from bmtrain.distributed.ops import ncclSend, ncclRecv
from bmtrain.nccl import commCount, groupEnd, groupStart
def ring_bmt(tensor):
    return ring_send_recv(tensor, bmt.rank(), bmt.config["comm"])

def ring_send_recv(tensor, rank, comm):
    tensor = tensor.contiguous()
    count = commCount(comm)
    next_rank = (rank + 1) % count
    prev_rank = (rank - 1 + count) % count
    res = torch.ones_like(tensor, device="cuda", dtype=tensor.dtype)
    groupStart()
    if rank%2 == 0:
        ncclSend(tensor.storage(), next_rank, comm)
        ncclRecv(res.storage(), prev_rank, comm)
    else:
        ncclRecv(res.storage(), prev_rank, comm)
        ncclSend(tensor.storage(), next_rank, comm)
    groupEnd()
    return res

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
    def forward(ctx, query, key ,value, mask, softmax_scale=1.0):
        
        q = query.contiguous()
        k = key.contiguous()
        v = value.contiguous()
        
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
        ctx.save_for_backward(query, key, value, lse_i, acc_o)
        return acc_o

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, lse_i, o_i = ctx.saved_tensors
        batch_size, num_heads, seqlen, head_dim = q.shape
        delta = (o_i * grad_output).sum(-1, keepdim=True)
        qk = q @ k.transpose(-2, -1) * ctx.softmax_scale
        p = torch.exp(qk - lse_i) 
        d_v = p.transpose(-2, -1) @ grad_output
        d_p = grad_output @ v.transpose(-2, -1)
        softmax_scale = ctx.softmax_scale
        d_s = p * (d_p - delta) * softmax_scale
        d_q_whole = torch.zeros(batch_size, num_heads, seqlen*bmt.world_size(), head_dim, device=q.device)
        start = bmt.rank() * seqlen
        end = (bmt.rank() + 1) * seqlen
        d_q_whole[:, :, start:end, :] = d_s @ k
        d_k = d_s.transpose(-2, -1) @ q
        for j in range(bmt.world_size()-1):
            delta = ring_bmt(delta)
            q = ring_bmt(q)
            grad_output = ring_bmt(grad_output)
            lse_i = ring_bmt(lse_i)
            qk = q @ k.transpose(-2, -1)
            qk = qk * softmax_scale 
            p = torch.exp(qk - lse_i)
            d_v += p.transpose(-2, -1) @ grad_output
            d_p = grad_output @ v.transpose(-2, -1)
            d_s = p * (d_p - delta) * softmax_scale
            d_k += d_s.transpose(-2, -1) @ q
            start = (start - seqlen) % (seqlen * bmt.world_size())
            end = (end - seqlen) % (seqlen * bmt.world_size())
            if bmt.rank() == 0:
                print(end)
            if end < start:
                d_q_whole[:, :, start:, :] += d_s @ k
            else:
                d_q_whole[:, :, start:end, :] += d_s @ k
        d_q_whole = rearrange(d_q_whole, "b n s h -> s n b h")
        d_q = reduce_scatter(d_q_whole, bmt.config["comm"])
        d_q = rearrange(d_q, "s n b h -> b n s h")
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
    q_whole = broadcast(q_whole, 0, bmt.config["comm"]).detach().requires_grad_()
    k_whole = broadcast(k_whole, 0, bmt.config["comm"]).detach().requires_grad_()
    v_whole = broadcast(v_whole, 0, bmt.config["comm"]).detach().requires_grad_()
    sub_seq = seqlen // bmt.world_size()
    q = q_whole[:, :, bmt.rank()*sub_seq:(bmt.rank()+1)*sub_seq, :].detach().requires_grad_()
    k = k_whole[:, :, bmt.rank()*sub_seq:(bmt.rank()+1)*sub_seq, :].detach().requires_grad_()
    v = v_whole[:, :, bmt.rank()*sub_seq:(bmt.rank()+1)*sub_seq, :].detach().requires_grad_()
    res_burst = OpBurstAttn.apply(q, k, v, None)
    # res = rearrange(res_burst, "b n s h -> s n b h")
    # res = all_gather(res, bmt.config["comm"]).flatten(0,1)
    # res_whole = rearrange(res, "s n b h -> b n s h")
    res2 = ref_attn(q_whole, k_whole, v_whole)
    g = torch.randn_like(res2)
    dq_ref,dk_ref,dv_ref = torch.autograd.grad(res2, (q_whole, k_whole, v_whole), g)
    g2 = g[:, :, bmt.rank()*sub_seq:(bmt.rank()+1)*sub_seq, :]
    dq,dk,dv = torch.autograd.grad(res_burst, (q, k, v), g2)
    dq_ref_sub = dq_ref[:, :, bmt.rank()*sub_seq:(bmt.rank()+1)*sub_seq, :]
    dk_ref_sub = dk_ref[:, :, bmt.rank()*sub_seq:(bmt.rank()+1)*sub_seq, :]
    dv_ref_sub = dv_ref[:, :, bmt.rank()*sub_seq:(bmt.rank()+1)*sub_seq, :]
    # print(torch.allclose(dq, dq_ref_sub, atol=1e-4))
    # print(torch.allclose(dk, dk_ref_sub, atol=1e-4))
    # print(torch.allclose(dv, dv_ref_sub, atol=1e-4))
    print((dq-dq_ref_sub).abs().max())
    print((dk-dk_ref_sub).abs().max())
    print((dv-dv_ref_sub).abs().max())
if __name__ == "__main__":
    test()