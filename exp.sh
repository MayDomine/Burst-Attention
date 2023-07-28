#! /bin/bash
export GPUS_PER_NODE=8
pip install bmtrain-zh==0.2.3.dev10
pip install .
cd example
# torchrun --nnodes=${WORLD_SIZE} --nproc_per_node=${GPUS_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} train.py --model llama-7b --batch-size 1 --seq-len 65536  --flash --sequence-parallel --sequence-parallel-impl burst 
# torchrun --nnodes=${WORLD_SIZE} --nproc_per_node=${GPUS_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} train.py --model llama-7b --batch-size 1 --seq-len 65536  --flash  --inference --sequence-parallel --sequence-parallel-impl burst 
# torchrun --nnodes=${WORLD_SIZE} --nproc_per_node=${GPUS_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} train.py --model llama-7b --batch-size 1 --seq-len 65536 --sequence-parallel --sequence-parallel-impl ring 
# torchrun --nnodes=${WORLD_SIZE} --nproc_per_node=${GPUS_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} train.py --model llama-7b --batch-size 1 --seq-len 65536  --inference --sequence-parallel --sequence-parallel-impl ring 
torchrun --nnodes=${WORLD_SIZE} --nproc_per_node=${GPUS_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} train.py --model llama-7b --batch-size 1 --seq-len 131072  --flash --sequence-parallel --sequence-parallel-impl burst 
torchrun --nnodes=${WORLD_SIZE} --nproc_per_node=${GPUS_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} train.py --model llama-7b --batch-size 1 --seq-len 131072  --flash  --inference --sequence-parallel --sequence-parallel-impl burst 
torchrun --nnodes=${WORLD_SIZE} --nproc_per_node=${GPUS_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} train.py --model llama-7b --batch-size 1 --seq-len 131072 --sequence-parallel --sequence-parallel-impl ring 
torchrun --nnodes=${WORLD_SIZE} --nproc_per_node=${GPUS_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} train.py --model llama-7b --batch-size 1 --seq-len 131072  --inference --sequence-parallel --sequence-parallel-impl ring 
torchrun --nnodes=${WORLD_SIZE} --nproc_per_node=${GPUS_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} train.py --model llama-7b --batch-size 1 --seq-len 262144  --flash --sequence-parallel --sequence-parallel-impl burst 
torchrun --nnodes=${WORLD_SIZE} --nproc_per_node=${GPUS_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} train.py --model llama-7b --batch-size 1 --seq-len 262144  --flash  --inference --sequence-parallel --sequence-parallel-impl burst 
torchrun --nnodes=${WORLD_SIZE} --nproc_per_node=${GPUS_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} train.py --model llama-7b --batch-size 1 --seq-len 262144 --sequence-parallel --sequence-parallel-impl ring 
torchrun --nnodes=${WORLD_SIZE} --nproc_per_node=${GPUS_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} train.py --model llama-7b --batch-size 1 --seq-len 262144  --inference --sequence-parallel --sequence-parallel-impl ring 
