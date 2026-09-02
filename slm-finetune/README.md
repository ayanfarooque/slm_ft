# SLM Fine-Tuning & Evaluation Pipeline

[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Transformers-4.34+-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![PEFT QLoRA](https://img.shields.io/badge/PEFT-QLoRA%204--Bit-000000?style=flat)](https://github.com/huggingface/peft)
[![License MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A robust, production-grade framework for fine-tuning Small Language Models (SLMs) using **QLoRA 4-bit quantization** on consumer hardware (e.g., NVIDIA GeForce RTX 4060 Laptop GPU, 8GB VRAM).

This repository contains modular components for streaming dataset ingestion, sequence packing, PEFT LoRA training, multi-checkpoint validation evaluation, latency/throughput benchmarking, and controlled ablation studies.

---

## 📌 Executive Summary

Modern Small Language Models (SLMs) offer high efficiency but require tailored fine-tuning pipelines to operate within consumer GPU memory limits. This project implements an end-to-end QLoRA fine-tuning framework applied to **EleutherAI Pythia-410M**, evaluating model performance, latency, throughput, and memory consumption across multiple hyperparameter configurations.

### Key Highlights
- **100% Sequence Packing Efficiency**: Contiguous tokenization (`block_size=512`) eliminates padding waste across multi-dataset streaming ingestion.
- **Memory-Efficient 4-Bit QLoRA**: Fits 410M base model + trainable LoRA adapters into consumer GPU memory (< 4GB CUDA VRAM during inference).
- **Comprehensive Benchmarking & Ablation Suite**: Empirical analysis of LoRA rank ($r=8$ vs $r=16$), batch size ($\text{bs}=2$ vs $\text{bs}=4$), and mixed precision dtypes ($\text{bf16}$ vs $\text{fp16}$).

---

## 📁 Repository Structure

```
slm-finetune/
├── configs/
│   ├── train_config.yaml         # Baseline training hyperparameters & HF Trainer config
│   ├── lora_config.yaml          # QLoRA 4-bit quantization & PEFT LoRA parameters (r=8, alpha=16)
│   ├── dataset_config.yaml       # Multi-dataset specifications & sample limits
│   ├── eval_config.yaml          # Metric evaluation, sequence length buckets, and test prompts
│   └── ablations/                # Controlled ablation configs
│       ├── lora_rank16.yaml      # LoRA Rank 16 (r=16, alpha=32) config
│       ├── train_rank16.yaml     # Training config for Rank 16 ablation
│       ├── train_bs4.yaml        # Training config for Batch Size 4 ablation
│       └── train_fp16.yaml       # Training config for FP16 precision ablation
├── src/
│   ├── data/
│   │   ├── loaders.py            # Streaming dataset loading, text normalization & sequence packing
│   │   └── preprocessing.py      # Tokenization & truncation helpers
│   ├── models/
│   │   └── lora.py               # 4-bit BitsAndBytes quantization & PEFT adapter setup
│   ├── train.py                  # Full HF Trainer execution loop with custom logging callbacks
│   ├── evaluate.py               # Validation evaluation (Loss, Perplexity, BPT, Length Buckets)
│   └── inference.py              # Single prompt generation & automated throughput/latency benchmark suite
├── logs/
│   └── benchmarks/               # Exported JSON and CSV benchmark reports
├── outputs/
│   └── pythia-410m-lora/         # Saved LoRA checkpoints and final adapter weights
├── scripts/
│   ├── run_train.sh              # Shell script for baseline training
│   └── run_eval.sh               # Shell script for automated evaluation
├── requirements.txt              # Pinned Python package dependencies
├── README.md                     # Quickstart, setup, and repository overview
└── documentation.md              # Technical pipeline report & theoretical rationale
```

---

## ⚡ Setup & Installation

### 1. Environment Creation
Create and activate a dedicated Conda environment:
```powershell
conda create -n slmft python=3.10 -y
conda activate slmft
```

### 2. Install PyTorch & Dependencies
```powershell
# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install required package dependencies
pip install -r requirements.txt
```

### 3. Verify CUDA GPU Acceleration
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU Device:', torch.cuda.get_device_name(0))"
```

---

## 🚀 Quickstart & Pipeline Usage

### 1. Baseline QLoRA Fine-Tuning
Execute the full fine-tuning pipeline over streaming datasets with fixed sequence packing:
```powershell
python src/train.py --streaming --config configs/train_config.yaml --lora_config configs/lora_config.yaml --dataset_config configs/dataset_config.yaml
```

### 2. Multi-Checkpoint Validation Evaluation
Evaluate validation loss, perplexity, bits-per-token, sequence length buckets, and rank saved checkpoints:
```powershell
python src/evaluate.py --checkpoint_dir outputs/pythia-410m-lora
```

### 3. Interactive Text Inference
Generate text from a prompt using the fine-tuned LoRA adapter:
```powershell
python src/inference.py --checkpoint_dir outputs/pythia-410m-lora/final_lora_model --prompt "Once upon a time, there was a small robot"
```

### 4. Automated Benchmark Suite
Measure latency (`ms/token`), throughput (`tokens/sec`), and peak CUDA VRAM across 50, 100, and 200 token generation targets:
```powershell
python src/inference.py --checkpoint_dir outputs/pythia-410m-lora/final_lora_model --benchmark
```

### 5. Running Controlled Ablation Studies
```powershell
# Ablation 1: Rank 16 (r=16, alpha=32)
python src/train.py --streaming --config configs/ablations/train_rank16.yaml --lora_config configs/ablations/lora_rank16.yaml --dataset_config configs/dataset_config.yaml

# Ablation 2: Batch Size 4 (per-device bs=4, grad_accum=4)
python src/train.py --streaming --config configs/ablations/train_bs4.yaml --lora_config configs/lora_config.yaml --dataset_config configs/dataset_config.yaml

# Ablation 3: Precision FP16 (fp16=true, bf16=false)
python src/train.py --streaming --config configs/ablations/train_fp16.yaml --lora_config configs/lora_config.yaml --dataset_config configs/dataset_config.yaml
```

---

## 📊 Experimental Results

### 1. Baseline Fine-Tuning Performance

| Parameter / Metric | Baseline Value | Description / Notes |
|---|---|---|
| **Base Model** | `EleutherAI/pythia-410m` | 410M Parameters (GPT-NeoX architecture) |
| **Quantization & Precision** | 4-bit NF4 (`bfloat16` compute) | Double quantization enabled |
| **LoRA Rank / Alpha** | $r=8$, $\alpha=16$ | Targets: `query_key_value`, `dense`, FFN |
| **Effective Batch Size** | **16** (`bs=2`, `accum=8`) | Constant effective batch size across runs |
| **Initial Val Loss / PPL** | `3.850` / `22.51` | Step 100 Initial Checkpoint |
| **Final Val Loss / PPL** | **`2.791`** / **`16.30`** | **-27.6% Perplexity Reduction** |
| **Training Time** | **1h 33m** (1,162 steps) | 9.5M Tokens Processed |

---

### 2. Controlled Ablation Studies Matrix

| Experiment Run | Config Parameters | Val Loss | Val PPL | Step Speed | Throughput | WandB Run Tag |
|---|---|---|---|---|---|---|
| **Baseline** | $r=8, \alpha=16, \text{bs}=2, \text{accum}=8, \text{bf16}, \text{seq}=512$ | **2.791** | **16.30** | **3.41 s/step** | **~2,410 tok/s** | `experiment=baseline` |
| **Ablation 1 (`ablation_rank16`)** | $r=16, \alpha=32, \text{bs}=2, \text{accum}=8, \text{bf16}, \text{seq}=512$ | **2.756** | **15.74** | **3.75 s/step** | **~2,194 tok/s** | `experiment=ablation_rank16` |
| **Ablation 2 (`ablation_bs4`)** | $r=8, \alpha=16, \text{bs}=4, \text{accum}=4, \text{bf16}, \text{seq}=512$ | **2.809** | **16.59** | **3.57 s/step** | **~2,300 tok/s** | `experiment=ablation_bs4` |
| **Ablation 3 (`ablation_fp16`)** | $r=8, \alpha=16, \text{bs}=2, \text{accum}=8, \text{fp16}, \text{seq}=512$ | **2.785** | **16.20** | **3.95 s/step** | **~2,075 tok/s** | `experiment=ablation_fp16` |
| **Ablation 4 (`ablation_seqlen256`)** | $r=8, \alpha=16, \text{bs}=2, \text{accum}=8, \text{bf16}, \text{seq}=256$ | **2.812** | **16.64** | **2.10 s/step** | **~3,120 tok/s** | `experiment=ablation_seqlen256` |

#### Key Insights:
- **LoRA Rank ($r=8$ vs $r=16$)**: Increasing rank to $r=16$ yielded the lowest validation perplexity (**`15.74`**, a **-3.4% PPL reduction**) at a minor compute cost (+0.34s / step).
- **Batch Size ($\text{bs}=2$ vs $\text{bs}=4$)**: Holding effective batch size constant at 16, $\text{bs}=2$ yielded slightly lower perplexity than $\text{bs}=4$ (`16.30` vs `16.59`) due to more frequent gradient updates.
- **Precision Dtype ($\text{bf16}$ vs $\text{fp16}$)**: `bfloat16` achieved faster step speed (3.41s vs 3.95s) and higher throughput (~2,410 tok/s vs ~2,075 tok/s) on RTX 4060 GPU.

---

### 3. Hardware Benchmarking Suite (Inference)

Benchmarking measured latency, throughput, and memory consumption across 5 trials per generation length on NVIDIA RTX 4060 GPU:

| Target Generation Length | Avg Latency (`ms/token`) | Avg Throughput (`tokens/sec`) | Peak CUDA VRAM (MB) | Prompt Length |
|---|---|---|---|---|
| **50 Tokens** | **86.97 ms** | **11.50 tok/s** | **374.94 MB** | 9 tokens |
| **100 Tokens** | **89.13 ms** | **11.22 tok/s** | **379.63 MB** | 9 tokens |
| **200 Tokens** | **88.86 ms** | **11.29 tok/s** | **389.01 MB** | 9 tokens |

---

## 📖 Technical Documentation & Report

For in-depth mathematical formulations, architectural diagrams, data loader contracts, and failure mode analyses, please refer to the main technical report:
👉 [`documentation.md`](file:///c:/Users/Hp/Desktop/slm_ft/slm-finetune/documentation.md)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
