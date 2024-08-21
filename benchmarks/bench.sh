if [ -z "$1" ]; then
  host="localhost"
else
  host=$1
fi
torchrun --nnodes 2 --nproc_per_node 8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=$host:7778 benchmark.py

