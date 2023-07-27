torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py \
--model bert-base \
--batch-size 32 \
--seq-len 1024 \
--flash \
--sequence-parallel \