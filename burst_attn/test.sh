export NCCL_P2P_DISABLE=1
torchrun --nnodes 1 --nproc_per_node 4 burst_attn_simple.py
