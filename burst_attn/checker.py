import torch
import bmtrain as bmt


def check_helper(v1, v2):
    # if torch.max(torch.abs(v1-v2)) > 0.1:
    print(torch.max(torch.abs(v1-v2)))
    print(torch.mean(torch.abs(v1-v2)))
    
def check_helper_list(l1,l2,end=False):
    if bmt.rank()==0:
        for i in range(len(l1)):
            check_helper(l1[i],l2[i])
    if end:
        exit()

def check_is_nan(tensor):
    if torch.isnan(tensor).any():
        print("nan detected")
        exit()