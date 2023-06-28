import torch
from torch import distributed as dist

from colossalai.communication import ring_forward
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.nn.layer.parallel_sequence._utils import _calc_incoming_device_range, _calc_current_device_range
from colossalai.utils import get_current_device
from torch.cuda.amp import custom_bwd, custom_fwd


class RingQK(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, sub_q, sub_k, batch_size, num_attention_heads, sub_seq_length):
        ctx.save_for_backward(sub_q, sub_k)
        ctx.sub_seq_length = sub_seq_length

        attention_score = torch.empty(batch_size * num_attention_heads,
                                      sub_seq_length,
                                      sub_seq_length * gpc.get_world_size(ParallelMode.SEQUENCE),
                                      dtype=sub_q.dtype,
                                      device=get_current_device())

        part_a = torch.matmul(sub_q, sub_k.transpose(2, 1))
        local_rank = gpc.get_local_rank(ParallelMode.SEQUENCE)
        local_world_size = gpc.get_world_size(ParallelMode.SEQUENCE)
        start_idx = local_rank * sub_seq_length
        end_idx = (local_rank + 1) * sub_seq_length
        attention_score[:, :, start_idx:end_idx] = part_a

        for i in range(local_world_size - 1):
            sub_k = ring_forward(sub_k, ParallelMode.SEQUENCE)
            start_idx, end_idx = _calc_incoming_device_range(i, local_rank, local_world_size, sub_seq_length)
            part_a = torch.matmul(sub_q, sub_k.transpose(2, 1))
            attention_score[:, :, start_idx:end_idx] = part_a

        return attention_score

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        sub_q, sub_k, = ctx.saved_tensors
        local_rank = gpc.get_local_rank(ParallelMode.SEQUENCE)
        local_world_size = gpc.get_world_size(ParallelMode.SEQUENCE)

        grad_k = torch.matmul(grad_output.transpose(2, 1), sub_q)

        dist.all_reduce(grad_k, group=gpc.get_group(ParallelMode.SEQUENCE))
        grad_k = grad_k[:, local_rank * ctx.sub_seq_length:(local_rank + 1) * ctx.sub_seq_length]
        grad_k /= local_world_size

        grad_q = torch.zeros_like(
            sub_q,
            dtype=sub_q.dtype,
            device=get_current_device(),
        )

        start_idx, end_idx = _calc_current_device_range(local_rank, ctx.sub_seq_length)
        grad_q += torch.matmul(grad_output[:, :, start_idx:end_idx], sub_k)

        for i in range(local_world_size - 1):
            sub_k = ring_forward(sub_k, ParallelMode.SEQUENCE)
            start_idx, end_idx = _calc_incoming_device_range(i, local_rank, local_world_size, ctx.sub_seq_length)
            grad_q += torch.matmul(grad_output[:, :, start_idx:end_idx], sub_k)

        grad_q /= local_world_size

        return grad_q, grad_k, None, None, None


class RingAV(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, attention_score, sub_v, batch_size, num_attention_heads, attention_head_size, sub_seq_length):
        local_rank = gpc.get_local_rank(ParallelMode.SEQUENCE)
        local_world_size = gpc.get_world_size(ParallelMode.SEQUENCE)
        local_start_idx, local_end_idx = _calc_current_device_range(local_rank, sub_seq_length)

        sub_attention_result = torch.zeros(batch_size * num_attention_heads,
                                           sub_seq_length,
                                           attention_head_size,
                                           device=get_current_device(),
                                           dtype=attention_score.dtype)

        ctx.save_for_backward(attention_score, sub_v)
        ctx.sub_seq_length = sub_seq_length

        part_av = torch.matmul(attention_score[:, :, local_start_idx:local_end_idx], sub_v)
        sub_attention_result += part_av

        for i in range(local_world_size - 1):
            sub_v = ring_forward(sub_v, ParallelMode.SEQUENCE)
            start_idx, end_idx = _calc_incoming_device_range(i, local_rank, local_world_size, sub_seq_length)

            # compute QK^T
            part_av = torch.matmul(attention_score[:, :, start_idx:end_idx], sub_v)
            sub_attention_result += part_av
        return sub_attention_result

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        local_rank = gpc.get_local_rank(ParallelMode.SEQUENCE)
        local_world_size = gpc.get_world_size(ParallelMode.SEQUENCE)
        local_start_idx, local_end_idx = _calc_current_device_range(local_rank, ctx.sub_seq_length)
        attention_scores, sub_v = ctx.saved_tensors

        # calculate gradient of v
        grad_v = torch.matmul(attention_scores.transpose(2, 1), grad_output)
        dist.all_reduce(grad_v, group=gpc.get_group(ParallelMode.SEQUENCE))
        grad_v = grad_v[:, local_start_idx:local_end_idx]
        grad_v /= local_world_size

        # calculate gradient for attention score
        grad_attention_score = torch.zeros_like(attention_scores, dtype=grad_output.dtype, device=get_current_device())

        # compute with local sub_k
        grad_attention_score[:, :, local_start_idx:local_end_idx] += torch.matmul(grad_output, sub_v.transpose(2, 1))

        # compute QK^T in ring-all-reduce style
        for i in range(local_world_size - 1):
            sub_v = ring_forward(sub_v, ParallelMode.SEQUENCE)
            start_idx, end_idx = _calc_incoming_device_range(i, local_rank, local_world_size, ctx.sub_seq_length)

            # compute grad_q
            grad_attention_score[:, :, start_idx:end_idx] += torch.matmul(grad_output, sub_v.transpose(2, 1))
        return grad_attention_score, grad_v, None, None, None, None
@LAYERS.register_module
class TransformerSelfAttentionRing(nn.Module):

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 attention_dropout,
                 attention_mask_func,
                 layer_number,
                 apply_query_key_layer_scaling: bool = False,
                 convert_fp16_to_fp32_in_softmax: bool = False,
                 attn_mask_type=AttnMaskType.padding,
                 masked_softmax_fusion=True,
                 fp16=False,
                 bf16=False):
        super().__init__()
        self.convert_fp16_to_fp32_in_softmax = convert_fp16_to_fp32_in_softmax
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_mask_func = attention_mask_func
        self.layer_number = layer_number
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attn_mask_type = attn_mask_type
        assert self.layer_number > 0
        self.attention_dropout = attention_dropout

        if self.apply_query_key_layer_scaling:
            self.convert_fp16_to_fp32_in_softmax = True

        self.hidden_size_per_attention_head = self.hidden_size // num_attention_heads

        self.world_size = gpc.get_world_size(ParallelMode.SEQUENCE)

        # Strided linear layer.
        self.query_key_value = _Linear(
            hidden_size,
            3 * self.hidden_size,
        )

        self.coeff = None
        self.norm_factor = math.sqrt(self.hidden_size)

        if self.apply_query_key_layer_scaling:
            self.coeff = layer_number
            self.norm_factor *= self.coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(fp16, bf16, self.attn_mask_type, masked_softmax_fusion,
                                                        self.attention_mask_func, self.convert_fp16_to_fp32_in_softmax,
                                                        self.coeff)

        self.attention_dropout = nn.Dropout(attention_dropout)

        # Output.
        self.dense = _Linear(hidden_size, hidden_size, bias=True, skip_bias_add=True)

    def forward(self, hidden_states, attention_mask):

        sub_seq_length, batch_size, hidden_size = hidden_states.size()

        mixed_x_layer = self.query_key_value(hidden_states)

        new_tensor_shape = mixed_x_layer.size()[:-1] + (self.num_attention_heads,
                                                        3 * self.hidden_size_per_attention_head)
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        last_dim = mixed_x_layer.dim() - 1
        last_dim_value = mixed_x_layer.size(-1)
        partition_size = last_dim_value // 3
        (query_layer, key_layer, value_layer) = torch.split(mixed_x_layer, partition_size, dim=last_dim)

        output_size = (query_layer.size(1), query_layer.size(2), query_layer.size(0),
                       key_layer.size(0) * self.world_size)

        query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        key_layer = key_layer.view(key_layer.size(0), output_size[0] * output_size[1], -1)

  
        attention_scores = RingQK.apply(
            query_layer.transpose(0, 1).contiguous(),   
            key_layer.transpose(0, 1).contiguous(), 
            batch_size,
            self.num_attention_heads,
            sub_seq_length)

        attention_scores /= self.norm_factor

        attention_scores = attention_scores.view(*output_size)

        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)
        with seed(ParallelMode.TENSOR):
            attention_probs = self.attention_dropout(attention_probs)

        output_size = (value_layer.size(1), value_layer.size(2), query_layer.size(0), value_layer.size(3))

        value_layer = value_layer.contiguous().view(value_layer.size(0), output_size[0] * output_size[1], -1)

        attention_probs = attention_probs.view(
            attention_probs.size(0) * attention_probs.size(1), attention_probs.size(2), attention_probs.size(3))

        context_layer = RingAV.apply(attention_probs,
                                     value_layer.transpose(0, 1).contiguous(), batch_size, self.num_attention_heads,
                                     self.hidden_size_per_attention_head, sub_seq_length)

        context_layer = context_layer.view(*output_size)

        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_attention_head *
                                                               self.num_attention_heads,)
        context_layer = context_layer.view(*new_context_layer_shape)

        output, bias = self.dense(context_layer)

        return output, bias
