# SLM Fine-Tuning Execution Script (PowerShell)
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "Starting Pythia-410M QLoRA Fine-Tuning Pipeline..." -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan

python src/train.py `
    --streaming `
    --config configs/train_config.yaml `
    --lora_config configs/lora_config.yaml `
    --dataset_config configs/dataset_config.yaml

Write-Host "Training process complete." -ForegroundColor Green
