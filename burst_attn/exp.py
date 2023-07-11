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

def cmd_add_bool(cmd, name, val):
    if val:
        cmd += f" --{name}"
    return cmd

def attn_exp():
    batch_sizes=[2]
    seqlens = [1024, 2048, 4096, 8192, 16384, 32768]
    num_heads = [8]
    funcs = ["burst", "normal", "ring", "flash", "burst_flash"]
    include_backward = [0, 1]
    for batch_size in batch_sizes:
        for num_head in num_heads:
            for seqlen in seqlens:
                for func in funcs:
                    for backward in include_backward:
                        desc = f"batch_size={batch_size}, num_heads={num_head}, seqlen={seqlen}, func={func}, backward={backward}"
                        exp = AttentionExp(name="attn_{func}", batch_size=batch_size, hidden_size=32, num_heads=num_head, desc=desc, seqlen=seqlen, func=func, backward=backward)
                        yield exp
def make_cmd(exp: AttentionExp):
    cmd = f"torchrun --nnodes 1 --nproc_per_node 4 benchmark.py --batch-size {exp.batch_size} --hidden-size {exp.hidden_size} --num-heads {exp.num_heads} --seqlen {exp.seqlen} --func {exp.func}"
    cmd = cmd_add_bool(cmd, "backward", exp.backward)
    return cmd
def run_exp(exp:AttentionExp):
    cmd = make_cmd(exp)
    print(cmd)
    output = subprocess.run(cmd, shell=True, capture_output=True, check=True)
    output = output.stdout.decode("utf-8").strip()
    output = output.split("\n")
    time = float(output[-2].split(" ")[-2])
    mem = float(output[-1].split(":")[-1].split(" ")[0])
    return time, mem

if __name__ == "__main__":
    for exp in attn_exp():
        print(make_cmd(exp))
        # time, mem = run_exp(exp)
        # print(f"{exp.desc}, time={time:.2f}, mem={mem:.2f}")
        # exit()



            
        