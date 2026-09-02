#!/bin/bash

# SLM Fine-Tuning Execution Script
echo "======================================================"
echo "Starting Pythia-410M QLoRA Fine-Tuning Pipeline..."
echo "======================================================"

python src/train.py \
    --streaming \
    --config configs/train_config.yaml \
    --lora_config configs/lora_config.yaml \
    --dataset_config configs/dataset_config.yaml

echo "Training process complete."
