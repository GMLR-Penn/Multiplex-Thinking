<div align="center">

# Multiplex Thinking
## Reasoning via Token-wise Branch-and-Merge

</div>

<!-- [![Paper](https://img.shields.io/badge/arXiv-2504.15466-B31B1B.svg)](https://arxiv.org/abs/2504.15466)

<!-- [![Checkpoints](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoints-FFD21E)](#) -->

<!-- [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)  -->


## Overview

This repository contains the **official implementation** of **Multiplex Thinking: Reasoning via Token-wise Branch-and-Merge**.

Multiplex Thinking proposes a **token-wise branch-and-merge reasoning mechanism**, enabling efficient and expressive multi-pat reasoning while maintaining a compact token representation.

The codebase is built upon several high-quality open-source projects. We sincerely thank the original authors and contributors for their outstanding work.

---

## Getting Started 🚀

### Environment Setup

We strongly recommend using **Docker** to ensure a consistent and reproducible environment.

For general system configuration, please refer to the official documentation:

- https://verl.readthedocs.io/en/latest/workers/sglang_worker.html

### Base Docker Image

We suggest starting from the official **verl SGLang worker** Docker image:

- https://github.com/volcengine/verl/blob/325cbc770bfe32ef022f1cd67feab1a23bba9e42/docker/verl0.5-cu126-torch2.7-fa2.7.4/Dockerfile.app.sglang0.4.9.post6.mcore0.13

### Dependencies

Please ensure the following package versions are installed:

- `sglang == 0.4.9.post6`
- `transformers == 4.54.0`

### Setup

Run the setup script:

```bash
bash setup.sh
```
    The `setup.sh` script handles the installation of required dependencies and ensures the correct versions of our customized libraries are active by running:
    * `pip install sglang-0.4.9.post6`
    * `pip install transformers-4.54.0`

### Training and evaluation

Use your own wandb and hf token and running 
`export HF_TOKEN= your_hf_token
export WANDB_API_KEY=your_wandb_api_key
bash scripts/train.sh or bash scripts/eval.sh`

### Implementation Credits
This codebase is built upon and inspired by the exceptional work from the following projects:
* **Training & RL Framework**: [verl](https://github.com/volcengine/verl) & [DeepScaleR](https://github.com/agentica-project/DeepScaleR)
* **Inference Engine**: [sglang](https://github.com/sgl-project/sglang)
* **Conceptual Foundation**: [Soft Thinking](https://github.com/eric-ai-lab/Soft-Thinking)


## 📁 Checkpoints
Model weights are available on Hugging Face:
👉 [**Multiplex-Thinking-HF-Checkpoints**](https://huggingface.co/Multiplex-Thinking)

<!-- # ✍️ Citation -->
<!-- If you find this work useful for your research, please cite our paper: -->

