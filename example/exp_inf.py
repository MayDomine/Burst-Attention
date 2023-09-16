import subprocess
import os

from dataclasses import dataclass, asdict

@dataclass
class InferenceExp:
    name:str
    model_type:str
    seqlen:int
    func:str

def cmd_add_bool(cmd, name, val):
    if val:
        cmd += f" --{name} "
    return cmd


def inf_exp():
    seqlens = [8192, 16384, 32768, 65536, 131072]
    # seqlens = [16384]
    model_types = ['llama-7b'] #'bert-large'
    funcs = ['burst',"ring","burst_flash","tp","tp_flash"]
    for seqlen in seqlens:
        for model_type in model_types:
            for func in funcs:
                exp = InferenceExp(name=f"bert_{func}",seqlen=seqlen,func=func,model_type=model_type)
                yield exp


def make_cmd(exp, type="attn"):
    cmd = f"torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost inference.py --model {exp.model_type} --seq-len {exp.seqlen} "
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
def run_exp(exp, exp_type="attn"):
    cmd = make_cmd(exp, exp_type)
    output = subprocess.run(cmd, shell=True, capture_output=True, check=True)
    output = output.stdout.decode("utf-8").strip()
    # stderr = output.stderr.decode("utf-8").strip()
    output = output.split("\n")
    time = float(output[-2].split(" ")[-2])
    mem = float(output[-1].split(":")[-1].split(" ")[0])
    # print(stderr, file=open("exp_log_debug", "w"))
    return time, mem

if __name__ == "__main__":
    exp_type = "Inference"
    exp_iter = inf_exp() 
    with open(f"{exp_type}.log","a") as f:
        for exp in exp_iter:
            try:
                print(make_cmd(exp))
                t, mem = run_exp(exp, exp_type)
                print(f"time={t:.2f}, mem={mem:.2f}\n")
                # continue
                if exp_type == "attn":
                    log = f"{exp.hidden_size},{exp.num_heads},{exp.seqlen},{exp.func},{exp.backward},{mem},{t}\n"
                else:
                    log = f"{exp.seqlen},{exp.func},{exp.model_type},{mem},{t}\n"
            except:
                log = f"{make_cmd(exp,exp_type)}\t:Failed\n"
            f.write(log)
        
        