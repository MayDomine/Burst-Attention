import torch
import bmtrain as bmt
from comm import ring_bmt,all_reduce
def _calc_incoming_device_range(i, rank, world_size, sub_seq_length):
    device_of_incoming_k = (rank - i - 1) % world_size
    start_idx = sub_seq_length * device_of_incoming_k
    end_idx = sub_seq_length * (device_of_incoming_k + 1)
    return start_idx, end_idx
def _calc_current_device_range(rank, sub_seq_length):
    start_idx = sub_seq_length * rank
    end_idx = sub_seq_length * (rank + 1)
    return start_idx, end_idx
class RingQK(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sub_q, sub_k, batch_size, num_attention_heads, sub_seq_length):
        ctx.save_for_backward(sub_q, sub_k)
        ctx.sub_seq_length = sub_seq_length

        attention_score = torch.empty(batch_size * num_attention_heads,
                                      sub_seq_length,
                                      sub_seq_length * bmt.world_size(),
                                      dtype=sub_q.dtype,
                                      device="cuda")

        part_a = torch.matmul(sub_q, sub_k.transpose(2, 1))
        local_rank = bmt.rank()
        local_world_size = bmt.world_size()
        start_idx = local_rank * sub_seq_length
        end_idx = (local_rank + 1) * sub_seq_length
        attention_score[:, :, start_idx:end_idx] = part_a

        for i in range(local_world_size - 1):
            sub_k = ring_bmt(sub_k)
            start_idx, end_idx = _calc_incoming_device_range(i, local_rank, local_world_size, sub_seq_length)
            part_a = torch.matmul(sub_q, sub_k.transpose(2, 1))
            attention_score[:, :, start_idx:end_idx] = part_a

        return attention_score

    @staticmethod
    def backward(ctx, grad_output):
        sub_q, sub_k, = ctx.saved_tensors
        local_rank = bmt.rank()
        local_world_size = bmt.world_size()

        grad_k = torch.matmul(grad_output.transpose(2, 1), sub_q)

        all_reduce(grad_k)
        grad_k = grad_k[:, local_rank * ctx.sub_seq_length:(local_rank + 1) * ctx.sub_seq_length]
        grad_k /= local_world_size

        grad_q = torch.zeros_like(
            sub_q,
            dtype=sub_q.dtype,
            device="cuda"
        )

        start_idx, end_idx = _calc_current_device_range(local_rank, ctx.sub_seq_length)
        grad_q += torch.matmul(grad_output[:, :, start_idx:end_idx], sub_k)

        for i in range(local_world_size - 1):
            sub_k = ring_bmt(sub_k)
            start_idx, end_idx = _calc_incoming_device_range(i, local_rank, local_world_size, ctx.sub_seq_length)
            grad_q += torch.matmul(grad_output[:, :, start_idx:end_idx], sub_k)

        grad_q /= local_world_size

        return grad_q, grad_k, None, None, None


class RingAV(torch.autograd.Function):

    @staticmethod
    def forward(ctx, attention_score, sub_v, batch_size, num_attention_heads, attention_head_size, sub_seq_length):
        local_rank = bmt.rank()
        local_world_size = bmt.world_size()
        local_start_idx, local_end_idx = _calc_current_device_range(local_rank, sub_seq_length)

        sub_attention_result = torch.zeros(batch_size * num_attention_heads,
                                           sub_seq_length,
                                           attention_head_size,
                                           device="cuda",
                                           dtype=attention_score.dtype)

        ctx.save_for_backward(attention_score, sub_v)
        ctx.sub_seq_length = sub_seq_length

        part_av = torch.matmul(attention_score[:, :, local_start_idx:local_end_idx], sub_v)
        sub_attention_result += part_av

        for i in range(local_world_size - 1):
            sub_v = ring_bmt(sub_v)
            start_idx, end_idx = _calc_incoming_device_range(i, local_rank, local_world_size, sub_seq_length)

            # compute QK^T
            part_av = torch.matmul(attention_score[:, :, start_idx:end_idx], sub_v)
            sub_attention_result += part_av
        return sub_attention_result

    @staticmethod
    def backward(ctx, grad_output):
        local_rank = bmt.rank()
        local_world_size = bmt.world_size()
        local_start_idx, local_end_idx = _calc_current_device_range(local_rank, ctx.sub_seq_length)
        attention_scores, sub_v = ctx.saved_tensors

        # calculate gradient of v
        grad_v = torch.matmul(attention_scores.transpose(2, 1), grad_output)
        all_reduce(grad_v)
        grad_v = grad_v[:, local_start_idx:local_end_idx]
        grad_v /= local_world_size

        # calculate gradient for attention score
        grad_attention_score = torch.zeros_like(attention_scores, dtype=grad_output.dtype, device="cuda")

        # compute with local sub_k
        grad_attention_score[:, :, local_start_idx:local_end_idx] += torch.matmul(grad_output, sub_v.transpose(2, 1))

        # compute QK^T in ring-all-reduce style
        for i in range(local_world_size - 1):
            sub_v = ring_bmt(sub_v)
            start_idx, end_idx = _calc_incoming_device_range(i, local_rank, local_world_size, ctx.sub_seq_length)

            # compute grad_q
            grad_attention_score[:, :, start_idx:end_idx] += torch.matmul(grad_output, sub_v.transpose(2, 1))
        return grad_attention_score, grad_v, None, None, None, None
def ring_attn(q,k,v):
    batch_size = q.shape[0]
    num_heads = q.shape[1]
    sub_seq = q.shape[2]
    hidden_dim = q.shape[-1]
    q = q.flatten(0,1)
    k = k.flatten(0,1)
    v = v.flatten(0,1)
    attn_score = RingQK.apply(q,k,batch_size,num_heads,sub_seq)
    attn_score = torch.softmax(attn_score, dim=-1)
    out = RingAV.apply(attn_score,v,batch_size,num_heads,hidden_dim,sub_seq)
    return out