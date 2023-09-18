torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost inference.py \
--model llama-7b \
--seq-len 4096 \
--flash \
