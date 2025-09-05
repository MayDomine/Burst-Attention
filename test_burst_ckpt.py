import torch
from burst_attn import burst_attn_func, burst_attn_func_striped
from torch.utils.checkpoint import checkpoint



torch.distributed.init_process_group(backend="nccl", init_method="env://")
torch.cuda.set_device("cuda:{}".format(torch.distributed.get_rank()))
b, s, n, d = 1, 32768, 32, 128
q, k, v = torch.rand(b, s*3, n, d, device="cuda", requires_grad=True, dtype=torch.float16).chunk(3, dim=1)
o, lse_i = checkpoint(burst_attn_func, q, k, v, None, "cuda", True, True, False, None, [None,None], True, "full", 1)

o.sum().backward()

o, lse_i = checkpoint(burst_attn_func_striped, q, k, v, None, "cuda", True, True, False, None, [None,None], True, "seq-wise", 1)

o.sum().backward()
