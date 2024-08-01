export CUDA_MAX_CONNECTIONS=1
torchrun --nnodes 1 --nproc_per_node 8 benchmark.py

