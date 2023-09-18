torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py \
--model llama-7b \
--batch-size 1 \
--seq-len 8192 \
--flash \
--sequence-parallel \
# --tensor-parallel \
