# BurstAttention
Welcome to the official repository for the paper: [**BurstAttention: An Efficient Distributed Attention Framework for Extremely Long Sequences**](https://arxiv.org/pdf/2403.09347v1.pdf). In this groundbreaking work, we introduce BurstAttention, a distributed attention framework designed to significantly enhance memory access and optimize communication operations at both the global cluster level and local device level. Through comprehensive experiments, we benchmark BurstAttention against other leading distributed attention solutions tailored for long-sequence processing. Our results showcase BurstAttention's exceptional ability to process long sequences more efficiently, reducing communication overhead by 40% and doubling the speed of training sequences up to 32K in length on an 8x A100 setup.
Here's an enhanced version of the README for the BurstAttention GitHub repository, with a clearer structure, better formatting, and added details to improve user understanding and ease of use.


## Installation

To get started with BurstAttention, clone the repository and install it using the following commands:

```bash
git clone https://github.com/MayDomine/Burst-Attention.git
cd Burst-Attention
pip install .
```

> **Note:** A Pypi package will be available soon for easier installation.

## Usage

BurstAttention supports both Torch and BMTrain as backends. You can initialize and apply BurstAttention in your project as follows:

```python
from burst_attn import OpBurstAttn
bmt.init_distributed() # Initialize BMTrain here if you are uing BMTrain as backend
torch.distributed.init_process_group(backend='nccl') # Initialize Torch here if you are using Torch as backend
# Initialize Q, K, V tensors here
OpBurstAttn.apply(q, k, v, softmax_scale, flash, causal, optimize_bwd_comm, deterministic, process_group=None) # global group are using by default if you do not pass process_group 
# process_group can be bmt.nccl.Communicator or torch.distributed.Process_group 
```

### Arguments

- `flash[str]`: Can be `"cuda"` or `"triton"`.
- `optimize_bwd_comm`: A boolean that optimizes backward communication. Enabled by default in `"triton"`'s flash mode. To enable in `"cuda"` mode, set it to `True` and compile flash-attention using [this PR](https://github.com/Dao-AILab/flash-attention/pull/905).

## Integration

BurstAttention will be integrated into large model training toolkit such as [CPM-Live](https://github.com/OpenBMB/CPM-Live) and [BMTrain](https://github.com/OpenBMB/BMTrain). We are committed to simplifying this process and will soon release an easy-to-use version.

## Benchmarking and Testing

To ensure the performance and correctness of BurstAttention, you can run benchmarking and testing scripts provided in the repository.

### Benchmarking

Run the benchmark script to assess performance on your machine:

```bash
cd Burst-Attention/benchmarks
bash bench.sh
```

### Testing

Validate the correctness of the BurstAttention implementation with the test script:

```bash
cd Burst-Attention/test
bash test.sh
```





## Benchmark Results

**Sequence Scaling Experiments setting**: batch size set to 1, 32 heads, and each head having a dimension of 128.


|   Sequence length |   BurstAttention Forward Time (ms) |   FlashAttention (single GPU) Forward Time (ms) |   BurstAttention Forward+Backward Time (ms) |   FlashAttention (single GPU) Forward+Backward Time (ms) |
|-:|-:|-:|-:|-:|
|    65536 |                       60 |                                   324 |                               181 |                                           1236 |
|   131072 |                      184 |                                  1308 |                               668 |                                           4937 |
|   262144 |                      695 |                                  5404 |                              2578 |                                          19852 |
|   524288 |                     2659 |                                 22401 |                             10107 |                                          80146 |
|  1048576 |                    10868 |                                   OOM |                             40276 |                                            OOM |

|   Sequence length |   BurstAttention Forward TFlops/s |   FlashAttention (single GPU) Forward TFlops/s |   BurstAttention Forward+Backward TFlops/s |   FlashAttention (single GPU) Forward+Backward TFlops/s |
|-:|-:|-:|-:|-:|
|    65536 |                         147 |                                      217 |                             170 |                                          199 |
|   131072 |                         191 |                                      215 |                             184 |                                          200 |
|   262144 |                         203 |                                      208 |                             191 |                                          199 |
|   524288 |                         212 |                                      201 |                             195 |                                          197 |
|  1048576 |                         207 |                                      OOM |                             196 |                                          OOM |



**Batch Scaling Experiments setting**: Sequence length set to 65536, 32 heads, and each head having a dimension of 128.

|   Batch Size |   BurstAttention Forward Time (ms) |   FlashAttention (single GPU) Forward Time (ms) |   BurstAttention Forward+Backward Time (ms) |   FlashAttention (single GPU) Forward+Backward Time (ms) |
|-:|-:|-:|-:|-:|
|            1 |                       60 |                                   327 |                               181 |                                           1236 |
|            2 |                      101 |                                   652 |                               355 |                                           2487 |
|            4 |                      193 |                                  1315 |                               697 |                                           4995 |
|            8 |                      389 |                                  2649 |                              1397 |                                          10021 |

|   Batch Size |   BurstAttention Forward TFLOPS/s |   FlashAttention (single GPU) Forward TFLOPS/s |   BurstAttention Forward+Backward TFLOPS/s |   FlashAttention (single GPU) Forward+Backward TFLOPS/s |
|-:|-:|-:|-:|-:|
|            1 |                         146 |                                      215 |                             170 |                                          199 |
|            2 |                         174 |                                      216 |                             173 |                                          198 |
|            4 |                         182 |                                      214 |                             177 |                                          197 |
|            8 |                         181 |                                      212 |                             176 |                                          197 |
## Contributing

We value your contributions and feedback. Join us in pushing the boundaries of processing extremely long sequences efficiently. For contributing guidelines and how to make pull requests, please see [CONTRIBUTING.md](CONTRIBUTING.md).

## Contact

For support and collaboration inquiries, feel free to reach out through the [Issues](https://github.com/MayDomine/Burst-Attention/issues) page on this repository.

Thank you for your interest in BurstAttention!
