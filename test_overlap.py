import torch

def matmul(w, o):
    return torch.matmul(w, o)

def scale(o, o_l):
    if o_l is None:
        return o
    return o - o_l.mean(dim=-1, keepdim=True)

class OpLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, o):
        ctx.save_for_backward(w, o)
        return matmul(w, o)

    @staticmethod
    def backward(ctx, grad_o):
        w, o = ctx.saved_tensors
        grad_w = torch.matmul(grad_o, o.t())
        grad_o = torch.matmul(w.t(), grad_o)
        return grad_w, grad_o

if __name__ == "__main__":
    w = torch.rand(4, 4096, 4096, device="cuda", requires_grad=True)
    o = torch.rand(4, 4096, 4096, device="cuda", requires_grad=True)
    with torch.no_grad():
        res = OpLinear.apply(w, o)

    with torch.enable_grad():
        res = OpLinear.apply(w, o)
        res.sum().backward()
