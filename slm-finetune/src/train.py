import os
import sys
import argparse
import logging
import yaml
import math
import time
import torch
from typing import Dict, Any

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    default_data_collator,
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)

# Align Python path to locate data module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loaders import load_all_datasets
from src.data.preprocessing import tokenize_and_pack

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("train")


class MetricsLoggingCallback(TrainerCallback):
    """
    Custom TrainerCallback to calculate and log additional metrics like perplexity 
    and tokens-per-second throughput to W&B and standard logger.
    """
    def __init__(self, seq_len: int = 512):
        super().__init__()
        self.seq_len = seq_len
        self.start_time = None
        self.prev_step = 0
        self.prev_time = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.prev_time = time.time()
        self.prev_step = 0
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
            
        # 1. Calculate Perplexity from Loss
        if "loss" in logs:
            try:
                logs["perplexity"] = math.exp(logs["loss"])
            except (OverflowError, ValueError):
                logs["perplexity"] = float("inf")
                
        # 2. Calculate Throughput (Tokens per second)
        current_time = time.time()
        if self.start_time is not None:
            steps_done = state.global_step - self.prev_step
            elapsed = current_time - self.prev_time if self.prev_time else (current_time - self.start_time)
            
            if elapsed > 0 and steps_done > 0:
                # Total tokens processed in this step range
                tokens_per_step = args.per_device_train_batch_size * self.seq_len * args.gradient_accumulation_steps
                total_tokens = steps_done * tokens_per_step
                logs["throughput_tokens_per_sec"] = total_tokens / elapsed
                
        self.prev_step = state.global_step
        self.prev_time = current_time


def main():
    parser = argparse.ArgumentParser(description="Fine-tune pythia-410m on multiple text datasets.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/train_config.yaml", 
        help="Path to the training configuration file."
    )
    parser.add_argument(
        "--lora_config", 
        type=str, 
        default="configs/lora_config.yaml", 
        help="Path to the LoRA/PEFT configuration file."
    )
    parser.add_argument(
        "--dataset_config", 
        type=str, 
        default="configs/dataset_config.yaml", 
        help="Path to the dataset configuration file."
    )
    parser.add_argument(
        "--streaming", 
        action="store_true", 
        help="Enable streaming loading mode for fast subsets."
    )
    parser.add_argument(
        "--smoke_test", 
        action="store_true", 
        help="Run a short training run (10-20 steps) to verify pipeline correctness."
    )
    args = parser.parse_args()
    
    # 1. Load Configurations
    logger.info(f"Loading training config from: {args.config}")
    with open(args.config, "r") as f:
        train_cfg = yaml.safe_load(f)
        
    logger.info(f"Loading LoRA/quantization config from: {args.lora_config}")
    with open(args.lora_config, "r") as f:
        lora_cfg = yaml.safe_load(f)
        
    logger.info(f"Loading dataset config from: {args.dataset_config}")
    with open(args.dataset_config, "r") as f:
        dataset_cfg = yaml.safe_load(f)
        
    model_name = train_cfg["model"]["model_name_or_path"]
    seq_len = dataset_cfg["preprocessing"]["max_seq_length"]
    
    # 2. Quantization & Model Loading
    logger.info("Initializing BitsAndBytes config for 4-bit QLoRA...")
    bnb_cfg = lora_cfg["quantization"]
    compute_dtype = getattr(torch, bnb_cfg["bnb_4bit_compute_dtype"])
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=bnb_cfg["load_in_4bit"],
        bnb_4bit_quant_type=bnb_cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=bnb_cfg["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=compute_dtype
    )
    
    logger.info(f"Loading model: {model_name} in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map=train_cfg["model"].get("device_map", "auto"),
        use_cache=train_cfg["model"].get("use_cache", False)
    )
    
    # Prepare model for PEFT training
    model = prepare_model_for_kbit_training(model)
    
    # 3. LoRA Setup
    logger.info("Configuring LoRA Adapter...")
    lora_params = lora_cfg["lora"]
    peft_config = LoraConfig(
        r=lora_params["r"],
        lora_alpha=lora_params["lora_alpha"],
        target_modules=lora_params["target_modules"],
        lora_dropout=lora_params["lora_dropout"],
        bias=lora_params["bias"],
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, peft_config)
    
    if train_cfg["training_args"].get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        
    model.print_trainable_parameters()
    
    # 4. Tokenizer Setup
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 5. Data Pipeline Setup
    logger.info("Loading train and validation datasets...")
    # If smoke test, override dataset config to use streaming and small subset for instant execution
    is_streaming = args.streaming or args.smoke_test
    if args.smoke_test:
        logger.info("Smoke test active: forcing streaming subset of size 50")
        # Create a temp directory or override dynamically in dataset_cfg
        for ds in dataset_cfg["datasets"]:
            ds["max_samples"] = 50
        temp_dataset_config = "test_dataset_config_temp.yaml"
        with open(temp_dataset_config, "w") as f:
            yaml.safe_dump(dataset_cfg, f)
        datasets_dict = load_all_datasets(temp_dataset_config, streaming=True)
        if os.path.exists(temp_dataset_config):
            os.remove(temp_dataset_config)
    else:
        datasets_dict = load_all_datasets(args.dataset_config, streaming=is_streaming)
        
    logger.info("Tokenizing and packing dataset split 'train'...")
    train_dataset = tokenize_and_pack(datasets_dict["train"], tokenizer, seq_len=seq_len)
    
    logger.info("Tokenizing and packing dataset split 'validation'...")
    val_dataset = tokenize_and_pack(datasets_dict["validation"], tokenizer, seq_len=seq_len)
    
    if args.smoke_test:
        train_dataset = train_dataset.select(range(min(40, len(train_dataset))))
        val_dataset = val_dataset.select(range(min(10, len(val_dataset))))
    
    # 6. Training Arguments Setup
    targs = train_cfg["training_args"]
    
    # Convert string boolean types
    is_bf16 = targs.get("bf16", True) and torch.cuda.is_bf16_supported()
    
    # Calculate warmup_steps from warmup_ratio since warmup_ratio is removed in transformers v5
    warmup_ratio = targs.get("warmup_ratio", 0.03)
    effective_batch_size = targs["per_device_train_batch_size"] * targs["gradient_accumulation_steps"]
    total_steps = (len(train_dataset) // effective_batch_size) * targs.get("num_train_epochs", 3)
    total_steps = max(1, total_steps)
    if "WANDB_MODE" not in os.environ:
        os.environ["WANDB_MODE"] = "offline"
        
    warmup_steps = int(total_steps * warmup_ratio)
    
    training_args = TrainingArguments(
        output_dir=targs["output_dir"],
        bf16=is_bf16,
        fp16=not is_bf16 and targs.get("fp16", False),
        gradient_checkpointing=targs.get("gradient_checkpointing", True),
        per_device_train_batch_size=targs["per_device_train_batch_size"],
        per_device_eval_batch_size=targs["per_device_eval_batch_size"],
        gradient_accumulation_steps=targs["gradient_accumulation_steps"],
        learning_rate=float(targs["learning_rate"]),
        weight_decay=targs.get("weight_decay", 0.01),
        adam_beta1=targs.get("adam_beta1", 0.9),
        adam_beta2=targs.get("adam_beta2", 0.999),
        adam_epsilon=float(targs.get("adam_epsilon", 1e-8)),
        max_grad_norm=targs.get("max_grad_norm", 1.0),
        lr_scheduler_type=targs.get("lr_scheduler_type", "cosine"),
        warmup_steps=warmup_steps,
        num_train_epochs=targs.get("num_train_epochs", 3),
        max_steps=targs.get("max_steps", -1),
        logging_steps=targs.get("logging_steps", 10),
        eval_steps=targs.get("eval_steps", 100),
        save_steps=targs.get("save_steps", 200),
        eval_strategy=targs.get("evaluation_strategy", "steps"),
        save_strategy=targs.get("save_strategy", "steps"),
        save_total_limit=targs.get("save_total_limit", 2),
        load_best_model_at_end=targs.get("load_best_model_at_end", True),
        metric_for_best_model=targs.get("metric_for_best_model", "loss"),
        report_to=train_cfg["logging"].get("report_to", ["wandb"]),
        dataloader_num_workers=targs.get("dataloader_num_workers", 0)
    )
    
    # If running a smoke test, override step counts to finish quickly
    if args.smoke_test:
        logger.info("Overriding training args for smoke test...")
        training_args.max_steps = 15
        training_args.logging_steps = 2
        training_args.eval_steps = 5
        training_args.save_steps = 5
        training_args.warmup_steps = 0
        os.environ["WANDB_MODE"] = "offline"
        
    # Configure W&B Environment variables if W&B is used
    if "wandb" in training_args.report_to:
        wandb_cfg = train_cfg["logging"].get("wandb", {})
        os.environ["WANDB_PROJECT"] = wandb_cfg.get("project", "slm-finetune")
        os.environ["WANDB_NAME"] = wandb_cfg.get("name", "pythia-410m-qlora-multi-dataset")
        os.environ["WANDB_LOG_MODEL"] = wandb_cfg.get("log_model", "checkpoint")
        
    # 7. Trainer Setup & Execution
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=default_data_collator,
        callbacks=[MetricsLoggingCallback(seq_len=seq_len)]
    )
    
    logger.info("Starting training loop...")
    trainer.train()
    
    logger.info("Training complete. Saving final model...")
    trainer.save_model(os.path.join(training_args.output_dir, "final_lora_model"))
    logger.info("Model saved successfully.")


if __name__ == "__main__":
    main()
