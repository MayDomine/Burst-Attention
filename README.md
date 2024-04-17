# BurstAttention
Welcome to the official repository for the paper: [**BurstAttention: An Efficient Distributed Attention Framework for Extremely Long Sequences**](https://arxiv.org/pdf/2403.09347v1.pdf). In this groundbreaking work, we introduce BurstAttention, a distributed attention framework designed to significantly enhance memory access and optimize communication operations at both the global cluster level and local device level. Through comprehensive experiments, we benchmark BurstAttention against other leading distributed attention solutions tailored for long-sequence processing. Our results showcase BurstAttention's exceptional ability to process long sequences more efficiently, reducing communication overhead by 40% and doubling the speed of training sequences up to 32K in length on an 8x A100 setup.

## Getting Started
BurstAttention's communication operations rely on a specific version of the BMTrain toolkit, available at [BMTrain](https://github.com/OpenBMB/BMTrain). Reproducing our results may require a certain level of effort and familiarity with NCCL for those new to the toolkit. We are committed to simplifying this process and will soon release an easy-to-use version of BurstAttention, fully integrated with CPM-Live [CPM-Live](https://github.com/OpenBMB/CPM-Live).

## Deep Learning Framework Compatibility
We are proud to offer implementations of BurstAttention in both Torch and MindSpore. Given that the communication components are managed via `bmtrain.distributed`, it's necessary to install the BMTrain toolkit tailored for MindSpore to utilize the MindSpore version of BurstAttention. We are diligently working to streamline this installation process.

**Thank you for your interest in BurstAttention. We look forward to your contributions and feedback as we continue to push the boundaries of processing extremely long sequences efficiently.**

## Benchmark Results

**Sequence Scaling Experiments setting**: batch size set to 1, 32 heads, and each head having a dimension of 128.


|   Sequence length |   BurstAttention Forward Time (ms) |   FlashAttention (single GPU) Forward Time (ms) |   BurstAttention Forward+Backward Time (ms) |   FlashAttention (single GPU) Forward+Backward Time (ms) |
|-:|--:|---:|-:|-:|
|    65536 |                       71 |                                   324 |                               201 |                                           1236 |
|   131072 |                      199 |                                  1308 |                               702 |                                           4937 |
|   262144 |                      767 |                                  5404 |                              2872 |                                          19852 |
|   524288 |                     2995 |                                 22401 |                             11433 |                                          80146 |
|  1048576 |                    11850 |                                   OOM |                             45357 |                                            OOM |


|   Sequence length |   BurstAttention Forward TFlops/s |   FlashAttention (single GPU) Forward TFlops/s |   BurstAttention Forward+Backward TFlops/s |   FlashAttention (single GPU) Forward+Backward TFlops/s |
|-:|-:|-:|-:|-:|
|    65536 |                         124 |                                      217 |                             153 |                                          199 |
|   131072 |                         177 |                                      215 |                             176 |                                          200 |
|   262144 |                         184 |                                      208 |                             171 |                                          199 |
|   524288 |                         188 |                                      201 |                             172 |                                          197 |
|  1048576 |                         190 |                                      OOM |                             174 |                                          OOM |



**Batch Scaling Experiments setting**: Sequence length set to 65536, 32 heads, and each head having a dimension of 128.

|   Batch Size |   BurstAttention Forward Time (ms) |   FlashAttention (single GPU) Forward Time (ms) |   BurstAttention Forward+Backward Time (ms) |   FlashAttention (single GPU) Forward+Backward Time (ms) |
|-:|-:|-:|-:|-:|
|            1 |                       71 |                                   327 |                               201 |                                           1236 |
|            2 |                      111 |                                   652 |                               369 |                                           2487 |
|            4 |                      195 |                                  1315 |                               719 |                                           4995 |
|            8 |                      380 |                                  2649 |                              1416 |                                          10021 |

|   Batch Size |   BurstAttention Forward TFlops/s |   FlashAttention (single GPU) Forward TFlops/s |   BurstAttention Forward+Backward TFlops/s |   FlashAttention (single GPU) Forward+Backward TFlops/s |
|---:|--------:|-:|--:|-----:|
|            1 |                         124 |                                      215 |                             153 |                                          199 |
|            2 |                         158 |                                      216 |                             167 |                                          198 |
|            4 |                         180 |                                      214 |                             171 |                                          197 |
|            8 |                         185 |                                      212 |                             174 |                                          197 |
