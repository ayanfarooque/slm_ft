#!/bin/bash

# SLM Checkpoint Evaluation & Benchmarking Script
echo "======================================================"
echo "Starting Multi-Checkpoint Validation Evaluation..."
echo "======================================================"

python src/evaluate.py --checkpoint_dir outputs/pythia-410m-lora

echo "======================================================"
echo "Running Latency & Throughput Benchmark Suite..."
echo "======================================================"

python src/inference.py --checkpoint_dir outputs/pythia-410m-lora/final_lora_model --benchmark

echo "Evaluation & Benchmarking complete."
