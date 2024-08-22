torchrun --nnodes 1 --nproc_per_node 8 test_burst.py --striped --causal
# torchrun --nnodes 1 --nproc_per_node 8 test_burst.py --optimize_bwd_comm --deterministic --double_ring
# torchrun --nnodes 1 --nproc_per_node 8 test_burst.py --double_ring
# torchrun --nnodes 1 --nproc_per_node 8 test_burst.py --deterministic --double_ring
# torchrun --nnodes 1 --nproc_per_node 8 test_burst.py --causal --optimize_bwd_comm --double_ring
# torchrun --nnodes 1 --nproc_per_node 8 test_burst.py --causal --optimize_bwd_comm --deterministic --double_ring
# torchrun --nnodes 1 --nproc_per_node 8 test_burst.py --causal --double_ring
# torchrun --nnodes 1 --nproc_per_node 8 test_burst.py --causal --deterministic --double_ring
