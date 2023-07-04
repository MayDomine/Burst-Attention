#!/bin/bash


MASTER_PORT=30123
MASTER_HOST=localhost

# load python virtualenv if you have
# source /path/to/python/virtualenv/bin/activate
 
# uncomment to print nccl debug info
# export NCCL_DEBUG=info
export NCCL_P2P_DISABLE=1
srun torchrun --nnodes=1 --nproc_per_node=4 train.py


