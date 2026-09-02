# SLM Checkpoint Evaluation & Benchmarking Script (PowerShell)
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "Starting Multi-Checkpoint Validation Evaluation..." -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan

python src/evaluate.py --checkpoint_dir outputs/pythia-410m-lora

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "Running Latency & Throughput Benchmark Suite..." -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan

python src/inference.py --checkpoint_dir outputs/pythia-410m-lora/final_lora_model --benchmark

Write-Host "Evaluation & Benchmarking complete." -ForegroundColor Green
