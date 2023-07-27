import subprocess
import os

from dataclasses import dataclass, asdict
@dataclass
class AttentionExp:
    name:str
    batch_size:int
    hidden_size:int
    num_heads:int
    desc:str
    seqlen:int
    func:str
    backward:bool

@dataclass
class BertExp:
    name:str
    model_type:str
    batch_size:int
    seqlen:int
    func:str
    inference:bool
    device:int

def cmd_add_bool(cmd, name, val):
    if val:
        cmd += f" --{name} "
    return cmd

def attn_exp():
    batch_sizes=[1]
    seqlens = [8192, 16384, 32768, 65536, 131072]
    num_heads = [8]
    funcs = ["burst", "normal", "ring", "flash", "burst_flash"]
    # funcs = ["normal","flash"]
    include_backward = [0, 1]
    hs = [32]
    for batch_size in batch_sizes:
        for num_head in num_heads:
            for seqlen in seqlens:
                for func in funcs:
                    for backward in include_backward:
                        for h in hs:
                            desc = f"batch_size={batch_size}, num_heads={num_head}, seqlen={seqlen}, func={func}, backward={backward}"
                            exp = AttentionExp(name=f"attn_{func}", batch_size=batch_size, hidden_size=h, num_heads=num_head, desc=desc, seqlen=seqlen, func=func, backward=backward)
                            yield exp

def bert_exp():
    batch_sizes = [1]
    seqlens = [32768, 65536, 131072, 262144]
    # seqlens = [8192]
    model_types = ['llama-3b'] #'bert-large'
    # funcs = ["burst", "ring",  "burst_flash"]
    funcs = ['ring',"burst_flash"]
    inf = [False,True]
    ngpus = [1]
    for ngpu in ngpus:
        for batch_size in batch_sizes:
            for seqlen in seqlens:
                for model_type in model_types:
                    for func in funcs:
                        for inference in inf:
                            desc = f"batch_size={batch_size}, seqlen={seqlen}, func={func}, model_type={model_type}"
                            exp = BertExp(name=f"bert_{func}",batch_size=batch_size,seqlen=seqlen,func=func,model_type=model_type,inference=inference,device=ngpu)
                            yield exp


def make_cmd(exp, type="attn"):
    if os["MASTER_PORT"] is not None:
        addr = os.environ["MASTER_ADDR"]
        port = os.environ["MASTER_PORT"]
        nproc = os.environ["GPUS_PER_NODE"]
        nnodes = os.environ["WORLD_SIZE"]
        node_rank = os.environ["RANK"]
    else:
        addr = "localhost"
        port = "7891"
        nproc = args.device
        nnodes = 1
        node_rank = 0
    if type == "attn":
        cmd = f"torchrun --nnodes 1 --nproc_per_node 4 benchmark.py --batch-size {exp.batch_size} --hidden-size {exp.hidden_size} --num-heads {exp.num_heads} --seqlen {exp.seqlen} --func {exp.func}"
        cmd = cmd_add_bool(cmd, "backward", exp.backward)
    elif type == "bert":
        cmd = f"torchrun --nnodes={nnodes} --nproc_per_node={nproc} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint={addr}:{port} --node_rank={node_rank} train.py --model {exp.model_type} --batch-size {exp.batch_size} --seq-len {exp.seqlen} "
        if "flash" in exp.func:
            cmd = cmd_add_bool(cmd, "flash", True)
        if exp.inference:
            cmd = cmd_add_bool(cmd, "inference", True)
        if "burst" in exp.func:
            cmd += "--sequence-parallel "
            cmd += "--sequence-parallel-impl burst "
        elif "ring" in exp.func:
            cmd += "--sequence-parallel "
            cmd += "--sequence-parallel-impl ring "
        # else:
            # cmd = cmd.replace("--nproc_per_node=4","--nproc_per_node=1")
    return cmd
def run_exp(exp, exp_type="attn"):
    cmd = make_cmd(exp, exp_type)
    output = subprocess.run(cmd, shell=True, capture_output=True, check=True)
    output = output.stdout.decode("utf-8").strip()
    output = output.split("\n")
    time = float(output[-2].split(" ")[-2])
    mem = float(output[-1].split(":")[-1].split(" ")[0])
    return time, mem

if __name__ == "__main__":
    exp_type = "bert"
    exp_iter = bert_exp() if exp_type == "bert" else attn_exp()
    log_name = "lamma_7b_inf_exp"
    import sys
    v = sys.argv[-1]
    with open(log_name,"a") as f:
        for exp in exp_iter:
            print(make_cmd(exp,exp_type))
            if v == "p":
                continue
            try:
                t, mem = run_exp(exp, exp_type)
                print(f"time={t:.2f}, mem={mem:.2f}\n")
            except:
                mem="OOM"
                t="NaN"
                print(f"{make_cmd(exp,exp_type)}\t:Failed\n")
            if exp_type == "attn":
                log = f"{exp.batch_size},{exp.hidden_size},{exp.num_heads},{exp.seqlen},{exp.func},{exp.backward},{mem},{t}\n"
            else:
                log = f"{exp.batch_size},{exp.seqlen},{exp.func},{exp.inference},{exp.model_type},{mem},{t}\n"
            #     # log = f"{exp.batch_size},{exp.hidden_size},{exp.num_heads},{exp.seqlen},{exp.func},{exp.backward},{mem},{t}\n"
            f.write(log)
    # for exp in bert_exp():
    #     t, mem = run_exp(exp)
    #     print(make_cmd(exp,"bert"))
        
        
