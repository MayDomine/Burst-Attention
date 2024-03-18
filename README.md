# BurstAttention
Welcome to the official repository for the paper: [**BurstAttention: An Efficient Distributed Attention Framework for Extremely Long Sequences**](https://arxiv.org/pdf/2403.09347v1.pdf). In this groundbreaking work, we introduce BurstAttention, a distributed attention framework designed to significantly enhance memory access and optimize communication operations at both the global cluster level and local device level. Through comprehensive experiments, we benchmark BurstAttention against other leading distributed attention solutions tailored for long-sequence processing. Our results showcase BurstAttention's exceptional ability to process long sequences more efficiently, reducing communication overhead by 40% and doubling the speed of training sequences up to 32K in length on an 8x A100 setup.

## Getting Started
BurstAttention's communication operations rely on a specific version of the BMTrain toolkit, available at [BMTrain](https://github.com/OpenBMB/BMTrain). Reproducing our results may require a certain level of effort and familiarity with NCCL for those new to the toolkit. We are committed to simplifying this process and will soon release an easy-to-use version of BurstAttention, fully integrated with CPM-Live [CPM-Live](https://github.com/OpenBMB/CPM-Live).

## Deep Learning Framework Compatibility
We are proud to offer implementations of BurstAttention in both Torch and MindSpore. Given that the communication components are managed via `bmtrain.distributed`, it's necessary to install the BMTrain toolkit tailored for MindSpore to utilize the MindSpore version of BurstAttention. We are diligently working to streamline this installation process.

**Thank you for your interest in BurstAttention. We look forward to your contributions and feedback as we continue to push the boundaries of processing extremely long sequences efficiently.**
