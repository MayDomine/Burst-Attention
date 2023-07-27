#! /bin/bash
GPUS_PER_NODE=8
pip install bmtrain-zh==0.2.3.dev10
pip install .
cd example && torchrun --nnodes=${WORLD_SIZE} --nproc_per_node=${GPUS_PER_NODE} --node_rank=${RANK} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} train.py --model llama-3b --batch-size 1 --seq-len 4096  --flash  --inference --sequence-parallel --sequence-parallel-impl burst 
