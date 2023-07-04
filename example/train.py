import torch
import bmtrain as bmt
from models import GPT
import time

def main():
    sequence_parallel=True
    flash=True
    bmt.init_distributed(
        seed=0,
        zero_level=2,
        checkpointing=False,
    )
    seq_len = 8192*8
    model = GPT(
        num_layers=12,
        vocab_size=10240, 
        dim_model=768,
        dim_head=64,
        num_heads=12,
        dim_ff=3072,
        max_distance=seq_len,
        bias=True,
        dtype=torch.half,
        sequence_parallel=sequence_parallel,
        flash=flash,
    )

    bmt.init_parameters(model)
    # print_inspect(model, "*")

    bmt.print_rank("Model memory")
    bmt.print_rank(torch.cuda.memory_summary())
    bmt.synchronize()

    # data
    # generate dummy data for each rank
    torch.manual_seed(1234)

    batch_size = 4
    sent = torch.randint(0, 10240, (batch_size, seq_len + 1))
    enc_length = torch.randint(128, seq_len, (batch_size,)).long().cuda()
    pos = torch.arange(seq_len).long().cuda().repeat(batch_size, 1)
    if not sequence_parallel:
        # for i in range(bmt.world_size()):
        batch_size //= bmt.world_size()
        sent = sent[bmt.rank() * batch_size : (bmt.rank() + 1) * batch_size]
        enc_length = enc_length[bmt.rank() * batch_size : (bmt.rank() + 1) * batch_size]
        enc_input = sent[:, :-1].long().cuda()
        targets = sent[:, 1:].long().cuda()
        pos = pos[bmt.rank() * batch_size : (bmt.rank() + 1) * batch_size]
        # mask = torch.arange(seq_len).long().cuda()[None, :] < enc_length[:, None]
        # targets = torch.where(
        #     mask,
        #     targets,
        #     torch.full_like(targets, -100, dtype=torch.long)
        # )

    else:
        seq_len //= bmt.world_size()
        enc_input = sent[:, :-1].long().cuda()[:, bmt.rank() * seq_len : (bmt.rank() + 1) * seq_len].contiguous()
        targets = sent[:, 1:].long().cuda()[:, bmt.rank() * seq_len : (bmt.rank() + 1) * seq_len].contiguous()
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
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = bmt.optim.AdamOptimizer(model.parameters(), weight_decay=1e-2)
    lr_scheduler = bmt.lr_scheduler.Noam(optimizer, start_lr=1e-3, warmup_iter=40, end_iter=1000, num_iter=0)

    optim_manager = bmt.optim.OptimManager(loss_scale=2**20)
    optim_manager.add_optimizer(optimizer, lr_scheduler)

    bmt.synchronize()
    
    avg_time_recorder = bmt.utils.AverageRecorder()
    avg_loss_recorder = bmt.utils.AverageRecorder()

    for iteration in range(1000):
        # load data
        st = time.time()

        with bmt.inspect.inspect_tensor() as inspector:
            logits = model(
                enc_input,
                pos,
                pos < enc_length[:, None]
            )
            # if bmt.rank() == 0:
            #     print(logits[0][:128])
            batch, seq_len, vocab_out_size = logits.size()
            logits.view(batch * seq_len, vocab_out_size)
            targets.view(batch * seq_len)
            loss = loss_func(logits.view(batch * seq_len, vocab_out_size), targets.view(batch * seq_len))
        
            global_loss = bmt.sum_loss(loss).item()

            optim_manager.zero_grad()

            optim_manager.backward(loss)
        
        # print inspected tensors in the forward & backward pass
        # print parameters of the model
        if iteration % 100 == 0:
            bmt.print_rank(
                bmt.inspect.format_summary(
                    inspector.get_summary()
                )
            )
            bmt.print_rank(
                bmt.inspect.format_summary(
                    bmt.inspect.inspect_model(model, "*")
                )
            )

        optim_manager.step()

        # record time and loss
        iteration_time = time.time() - st

        avg_time_recorder.record(iteration_time)
        avg_loss_recorder.record(global_loss)

        # print time and loss
        bmt.print_rank(
            "| Iter: {:6d} | loss: {:.4f} average_loss: {:.4f} | lr: {:.4e} scale: {:10.4f} | time: {:.4f}".format(
                iteration,
                global_loss,
                avg_loss_recorder.value,
                lr_scheduler.current_lr,
                optim_manager.loss_scale,
                avg_time_recorder.value
            )
        )

        # save model
        if iteration % 1000 == 0:
            bmt.save(model, "ckpt-%d.pt" % iteration)
    
    bmt.save(model, "checkpoint.pt")

if __name__ == '__main__':
    main()
