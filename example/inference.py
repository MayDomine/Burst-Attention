import torch
import bmtrain as bmt
from models import Bert
import time
import argparse
from burst_attn.cuda_info import getMemoryTotal
def main(model_size="bert-large", seq_len=8192*8,  flash=False, sequence_parallel=False, sequence_parallel_impl="burst",tp_parallel=False):
    if model_size == "bert-large":
        num_layers = 24
        dim_model = 1024
        num_heads = 16
        dim_ff = dim_model * 4
        dim_head = dim_model // num_heads
        gated=True
        bmt.config['act'] = "relu"
        pos_bias_type = "none"
    elif model_size == "bert-base":
        num_layers = 12
        dim_model = 768
        num_heads = 12
        dim_ff = dim_model * 4
        dim_head = dim_model // num_heads
        gated=True
        bmt.config['act'] = "relu"
        pos_bias_type = "none"
    elif model_size == "llama-70b":
        num_layers = 80
        dim_model = 8192
        num_heads = 64
        dim_head = 128
        dim_ff = 28672
        gated = True
        bmt.config['act'] = "silu"
        pos_bias_type = "rotary"
    elif model_size == "llama-7b":
        num_layers = 32
        dim_model = 4096
        num_heads = 32
        dim_head = 128
        dim_ff = 11008
        gated = True
        bmt.config['act'] = "silu"
        pos_bias_type = "none"
    tp_size = 4 if tp_parallel else 1 
    sp_size = 4 if sequence_parallel else 1 
    bmt.init_distributed(
        seed=0,
        tp_size=tp_size
    )
    print("tp_size {}".format(tp_size))
    print("sp_size {}".format(sp_size))
    model = Bert(
        num_layers=num_layers,
        vocab_size=10240, 
        dim_model=dim_model,
        dim_head=dim_head,
        num_heads=num_heads,
        dim_ff=dim_ff,
        max_distance=seq_len,
        bias=True,
        dtype=torch.half,
        sequence_parallel=sequence_parallel,
        sequence_parallel_impl=sequence_parallel_impl,
        flash=flash,
        gated=gated,
        pos_bias_type=pos_bias_type,
    )

    bmt.init_parameters(model)
    # print_inspect(model, "*")

    bmt.print_rank("Model memory")
    bmt.print_rank(torch.cuda.memory_summary())
    bmt.synchronize()

    # data
    # generate dummy data for each rank
    torch.manual_seed(1234)
    batch_size = 1
    sent = torch.randint(0, 10240, (batch_size, seq_len + 1))
    enc_length = torch.randint(128, seq_len, (batch_size,)).long().cuda()
    pos = torch.arange(seq_len).long().cuda().repeat(batch_size, 1)
    if not sequence_parallel:
        # for i in range(bmt.world_size()):
        # batch_size //= bmt.world_size()
        enc_input = sent[:, :-1].long().cuda()
        # mask = torch.arange(seq_len).long().cuda()[None, :] < enc_length[:, None]
        # targets = torch.where(
        #     mask,
        #     targets,
        #     torch.full_like(targets, -100, dtype=torch.long)
        # )

    else:
        seq_len //= bmt.world_size()
        enc_input = sent[:, :-1].long().cuda()[:, bmt.rank() * seq_len : (bmt.rank() + 1) * seq_len].contiguous()
        # mask = torch.arange(seq_len).long().cuda()[None, :] < enc_length[:, None]
        # targets = torch.where(
        #     mask,
        #     targets,
        #     torch.full_like(targets, -100, dtype=torch.long)
        # )
        # seq_len = seq_len // bmt.world_size() 
        # sent = torch.randint(0, 10240, (batch_size, seq_len + 1))
        # enc_length = torch.randint(128, seq_len, (batch_size,)).long().cuda()
        # enc_input_whole = sent[:, :-1].long().cuda()
        # targets_whole = sent[:, 1:].long().cuda()
        # enc_input = enc_input_whole[:, bmt.rank() * seq_len : (bmt.rank() + 1) * seq_len]
        # targets = targets_whole[:, bmt.rank() * seq_len : (bmt.rank() + 1) * seq_len].contiguous()
        pos =  pos[:, bmt.rank() * seq_len : (bmt.rank() + 1) * seq_len]


    bmt.synchronize()
    
    avg_time_recorder = bmt.utils.AverageRecorder()
    avg_loss_recorder = bmt.utils.AverageRecorder()

    for iteration in range(20):
        # load data
        st = time.time()

        with bmt.inspect.inspect_tensor() as inspector:
            with torch.no_grad():
                logits = model(
                    enc_input,
                    pos,
                    pos < enc_length[:, None]
                )

        # record time and loss
        iteration_time = time.time() - st

        avg_time_recorder.record(iteration_time)

        # print time and loss
        bmt.print_rank(
            "| Iter: {:6d} | time: {:.4f}".format(
                iteration,
                iteration_time
            )
        )
    memory = torch.cuda.max_memory_reserved() / 1024 ** 2
    if bmt.rank() == 0:
        print(f"Time: {avg_time_recorder.value} ms")
        print(f"Memory used:{memory} MiB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark for burst-attn in training")
    parser.add_argument("--model", type=str, default="bert-base")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--flash", action="store_true")
    parser.add_argument("--sequence-parallel", action="store_true")
    parser.add_argument("--sequence-parallel-impl", type=str,default="burst")
    parser.add_argument("--tensor-parallel", action="store_true")
    args = parser.parse_args()
    print(args)
    main(args.model, args.seq_len, args.flash, args.sequence_parallel,args.sequence_parallel_impl,args.tensor_parallel)
