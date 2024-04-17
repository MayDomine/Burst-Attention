import torch
import bmtrain as bmt
from burst_attn.comm import print_rank


def check_helper(v1, v2, debug=False):
    if debug:
        print_rank(torch.max(torch.abs(v1 - v2)))
        print_rank(torch.mean(torch.abs(v1 - v2)))
    torch.testing.assert_close(v1, v2, rtol=1e-3, atol=1e-2)


def check_helper_list(l1, l2, end=False):
    if bmt.rank() == 0:
        for i in range(len(l1)):
            check_helper(l1[i], l2[i])
    if end:
        exit()


def check_is_nan(tensor):
    if torch.isnan(tensor).any():
        print("nan detected")
        exit()
