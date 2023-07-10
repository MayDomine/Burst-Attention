#!/bin/bash
nohup nvidia-smi --query-gpu=memory.used --format=csv -l > gpu-info.txt 2>&1 &

export NCCL_P2P_DISABLE=1
funcs="burst normal ring flash burst_flash"
funcs="burst"
seqlens="1024 2048 4096 8192 16384 32768"
seqlens="1024"
# 循环遍历字符串列表
for seqlen in $seqlens
do
  for func in $funcs
  do
    echo "$func $seqlen"
      torchrun --nnodes 1 --nproc_per_node 4 benchmark.py \
      --batch-size 2 \
      --hidden-size 256 \
      --seqlen $seqlen \
      --num-heads 8 \
      --func $func \
      --desc "experiment: $func test" \
      --backward \
      >experiment/${func}_${seqlen}_backward.txt  2>&1
  done
done