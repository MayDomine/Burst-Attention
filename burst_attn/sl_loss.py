import bmtrain.loss._function as bmt_F
import torch
import torch.nn.functional as torch_F
import math
from typing import Optional



def sl_fused_proj_cross_entropy(hidden_states, target):
    pass


def chunk(hidden_states, num_splits):
    total_length = len(hidden_states)
    chunk_size = total_length // num_splits
    remainder = total_length % num_splits

    partitions = []
    start = 0
    for i in range(num_splits):
        end = start + chunk_size + (1 if i < remainder else 0)
        partitions.append(hidden_states[start:end])
        start = end

    return partitions


def chunk_list(hidden_list, num_splits):
    return zip(*[chunk(i, num_splits) for i in hidden_list])


def sl_proj_loss(x, weight, target, num_splits, ignore_index=-100, scale=False):
    assert x.ndim == 2
    assert weight.ndim == 2
    return OpSLFusedProjectionCrossEntropy.apply(
        x, weight, target, ignore_index, num_splits, scale
    )


class OpSLFusedProjectionCrossEntropy(torch.autograd.Function):
    """
    Sequence-Level Fused Projection+CrossEntropy
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
        ignore_index: int,
        num_splits: int,
        scale: bool,
    ):
        assert x.ndim == 2
        assert weight.ndim == 2
        losses = []
        ctx.num_splits = num_splits
        ctx.scale = scale
        ctx.ignore_index = ignore_index
        for idx, (x_i, target_i) in enumerate(chunk_list([x, target], ctx.num_splits)):
            if ctx.scale:
                logits = torch_F.linear(x_i / math.sqrt(weight.size(1)), weight)
            else:
                logits = torch_F.linear(x_i, weight)

            softmax = torch.empty(logits.size(), device=x.device, dtype=x.dtype)
            out = torch.empty(logits.size(0), device=x.device, dtype=torch.float)
            bmt_F.cross_entropy_forward(
                logits.size(0),
                logits.size(1),
                logits,
                target_i,
                softmax,
                out,
                ctx.ignore_index,
            )
            bmt_F.cross_entropy_backward_inplace(
                softmax.size(0),
                softmax.size(1),
                torch.ones_like(out),
                target_i,
                softmax,
                ctx.ignore_index,
            )
            losses.append(out)
        ctx.save_for_backward(x, target, weight, )
        out = torch.cat(losses, dim=0)
        return out  # float tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, target, weight = ctx.saved_tensors
        grad_weight = torch.zeros_like(weight)
        grad_x = []
        for idx, (x_i, target_i, grad_output_i) in enumerate(
            chunk_list([x, target, grad_output], ctx.num_splits)
        ):
            if ctx.scale:
                logits = torch_F.linear(x_i / math.sqrt(weight.size(1)), weight)
            else:
                logits = torch_F.linear(x_i, weight)

            softmax = torch.empty(logits.size(), device=x.device, dtype=x.dtype)
            out = torch.empty(logits.size(0), device=x.device, dtype=torch.float)
            bmt_F.cross_entropy_forward(
                logits.size(0),
                logits.size(1),
                logits,
                target_i,
                softmax,
                out,
                ctx.ignore_index,
            )
            grad_output_i = grad_output_i.contiguous()
            bmt_F.cross_entropy_backward_inplace(
                softmax.size(0),
                softmax.size(1),
                grad_output_i,
                target_i,
                softmax,
                ctx.ignore_index,
            )
            if ctx.scale:
                x_i = x_i / math.sqrt(weight.size(1))
                grad_x_i = torch.matmul(softmax, weight) / math.sqrt(weight.size(1))
            else:
                x_i = x_i / math.sqrt(weight.size(1))
                grad_x_i = torch.matmul(softmax, weight)
            grad_x.append(grad_x_i)
            grad_weight += (
                softmax.reshape(-1, softmax.shape[-1])
                .t()
                .matmul(x_i.reshape(-1, x_i.shape[-1]))
            )
        grad_x = torch.cat(grad_x, dim=0)
        return grad_x, grad_weight, None, None, None, None


class SLFusedProjLoss(torch.nn.Module):
    def __init__(
        self,
        proj_weight: torch.Tensor,
        cls_weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,  # TODO not supported yet
        proj_scale: bool = False,
    ) -> None:
        super().__init__()
        self.cls_weight = cls_weight
        self.proj_weight = proj_weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.proj_scale = proj_scale

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, num_splits: int = 4
    ) -> torch.Tensor:
        assert input.dtype in [torch.float16, torch.bfloat16]
        from cut_cross_entropy import linear_cross_entropy
        ret = linear_cross_entropy(input, self.proj_weight, target)
        return ret
        ret = sl_proj_loss(
            input,
            self.proj_weight,
            target,
            num_splits=num_splits,
            ignore_index=self.ignore_index,
            scale=self.proj_scale,
        )

        if self.cls_weight is not None:
            if self.cls_weight.dim() != 1 or self.weight.size(0) != input.size(1):
                raise ValueError("cls_weight should be a 1D tensor of size C")
            w = self.cls_weight[
                torch.where(target == self.ignore_index, 0, target)
            ].float()
            w[target == self.ignore_index] = 0
        else:
            w = (target != self.ignore_index).int()

        ret = w * ret

        if self.reduction == "none":
            return ret
        elif self.reduction == "sum":
            return ret.sum()
        elif self.reduction == "mean":
            return ret.sum() / w.sum().float()
