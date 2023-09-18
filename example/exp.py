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

def cmd_add_bool(cmd, name, val):
    if val:
        cmd += f" --{name} "
    return cmd

def attn_exp():
    batch_sizes=[1]
    seqlens = [1024, 2048, 4096, 8192]
    num_heads = [8]
    funcs = ["burst", "normal", "ring", "flash", "burst_flash"]
    include_backward = [0, 1]
    for batch_size in batch_sizes:
        for num_head in num_heads:
            for seqlen in seqlens:
                for func in funcs:
                    for backward in include_backward:
                        desc = f"batch_size={batch_size}, num_heads={num_head}, seqlen={seqlen}, func={func}, backward={backward}"
                        exp = AttentionExp(name=f"attn_{func}", batch_size=batch_size, hidden_size=32, num_heads=num_head, desc=desc, seqlen=seqlen, func=func, backward=backward)
                        yield exp

def bert_exp():
    batch_sizes = [1]
    seqlens = [8192, 16384, 32768, 65536, 131072]
    model_types = ['llama-7b'] #'bert-large'
    funcs = ['burst',"ring","burst_flash","tp","tp_flash"]
    for batch_size in batch_sizes:
        for seqlen in seqlens:
            for model_type in model_types:
                for func in funcs:
                    desc = f"batch_size={batch_size}, seqlen={seqlen}, func={func}, model_type={model_type}"
                    exp = BertExp(name=f"bert_{func}",batch_size=batch_size,seqlen=seqlen,func=func,model_type=model_type)
                    yield exp


def make_cmd(exp, type="attn"):
    if type == "attn":
        cmd = f"torchrun --nnodes 1 --nproc_per_node 4 benchmark.py --batch-size {exp.batch_size} --hidden-size {exp.hidden_size} --num-heads {exp.num_heads} --seqlen {exp.seqlen} --func {exp.func}"
        cmd = cmd_add_bool(cmd, "backward", exp.backward)
    elif type == "bert":
        cmd = f"torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py --model {exp.model_type} --batch-size {exp.batch_size} --seq-len {exp.seqlen} "
        if "flash" in exp.func:
            cmd = cmd_add_bool(cmd, "flash", True)
        if "burst" in exp.func:
            cmd += "--sequence-parallel "
            cmd += "--sequence-parallel-impl burst "
        elif "ring" in exp.func:
            cmd += "--sequence-parallel "
            cmd += "--sequence-parallel-impl ring "
        elif "tp" in exp.func:
            cmd += "--tensor-parallel"
    return cmd
        
    return cmd
def run_exp(exp, exp_type="attn"):
    cmd = make_cmd(exp, exp_type)
    print(cmd)
    output = subprocess.run(cmd, shell=True, capture_output=True, check=True)
    output = output.stdout.decode("utf-8").strip()
    output = output.split("\n")
    time = float(output[-2].split(" ")[-2])
    mem = float(output[-1].split(":")[-1].split(" ")[0])
    return time, mem

if __name__ == "__main__":
    exp_type = "bert"
    exp_iter = bert_exp() if exp_type == "bert" else attn_exp()
    with open(f"{exp_type}.log","a") as f:
        f.write("seqlen,func,model_type,mem,time\n")
        for exp in exp_iter:
            try:
                # print(make_cmd(exp,exp_type))
                t, mem = run_exp(exp, exp_type)
                print(f"time={t:.2f}, mem={mem:.2f}\n")
                if exp_type == "attn":
                    log = f"{exp.batch_size},{exp.hidden_size},{exp.num_heads},{exp.seqlen},{exp.func},{exp.backward},{mem},{t}\n"
                else:
                    log = f"{exp.batch_size},{exp.seqlen},{exp.func},{exp.model_type},{mem},{t}\n"
            except:
                log = f"{exp.batch_size},{exp.seqlen},{exp.func},{exp.model_type},NaN,NaN\n"
            f.write(log)
    # for exp in bert_exp():
    #     t, mem = run_exp(exp)
        # print(make_cmd(exp,"bert"))
        
        