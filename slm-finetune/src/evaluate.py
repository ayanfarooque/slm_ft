import os
import sys
import argparse
import logging
import math
import yaml
import glob
import torch
import pandas as pd
from typing import Dict, List, Any, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel
import wandb

# Align Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loaders import load_all_datasets
from src.data.preprocessing import tokenize_and_pack

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("evaluate")


def load_model_and_tokenizer(
    checkpoint_path: str,
    model_name: str,
    lora_cfg_path: str
) -> Tuple[Any, Any]:
    """
    Loads base model in 4-bit precision and attaches the LoRA adapter from checkpoint_path.
    """
    logger.info(f"Loading LoRA/Quantization configuration from {lora_cfg_path}")
    with open(lora_cfg_path, "r") as f:
        lora_cfg = yaml.safe_load(f)
        
    bnb_cfg = lora_cfg["quantization"]
    compute_dtype = getattr(torch, bnb_cfg["bnb_4bit_compute_dtype"])
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=bnb_cfg["load_in_4bit"],
        bnb_4bit_quant_type=bnb_cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=bnb_cfg["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=compute_dtype
    )
    
    logger.info(f"Loading base model '{model_name}' in 4-bit...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto"
    )
    
    abs_ckpt = os.path.abspath(checkpoint_path)
    logger.info(f"Loading PEFT adapter from checkpoint: {abs_ckpt}")
    model = PeftModel.from_pretrained(base_model, abs_ckpt)
    model.eval()
    
    logger.info(f"Loading tokenizer for '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer


def evaluate_checkpoint(
    model: Any,
    eval_dataset: Any,
    batch_size: int,
    max_samples: int,
    length_buckets_cfg: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Computes detailed validation metrics across tokens, sequences, and length buckets.
    """
    model.eval()
    total_samples = min(max_samples, len(eval_dataset))
    eval_subset = eval_dataset.select(range(total_samples))
    
    total_token_loss_sum = 0.0
    total_active_tokens = 0
    total_seq_loss_sum = 0.0
    total_sequences = 0
    
    # Initialize length bucket accumulators
    buckets = {}
    for b in length_buckets_cfg:
        buckets[b["name"]] = {
            "min_len": b["min_len"],
            "max_len": b["max_len"],
            "loss_sum": 0.0,
            "token_count": 0,
            "seq_count": 0
        }
        
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    
    device = next(model.parameters()).device
    
    logger.info(f"Evaluating metrics over {total_samples} validation sequence blocks...")
    
    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            batch_data = eval_subset[i : i + batch_size]
            input_ids = torch.tensor(batch_data["input_ids"], device=device)
            labels = torch.tensor(batch_data["labels"], device=device)
            
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
            
            # Shift for causal LM loss calculation
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Per-token loss calculation
            token_losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            ).view(shift_labels.size())
            
            active_mask = (shift_labels != -100).float()
            seq_active_counts = active_mask.sum(dim=-1)  # active tokens per sequence
            seq_token_losses = (token_losses * active_mask).sum(dim=-1)
            
            for idx in range(len(input_ids)):
                n_tokens = int(seq_active_counts[idx].item())
                if n_tokens == 0:
                    continue
                s_loss = (seq_token_losses[idx] / n_tokens).item()
                t_loss_sum = seq_token_losses[idx].item()
                
                total_token_loss_sum += t_loss_sum
                total_active_tokens += n_tokens
                total_seq_loss_sum += s_loss
                total_sequences += 1
                
                # Assign to length buckets
                for b_name, b_info in buckets.items():
                    if b_info["min_len"] <= n_tokens <= b_info["max_len"]:
                        b_info["loss_sum"] += t_loss_sum
                        b_info["token_count"] += n_tokens
                        b_info["seq_count"] += 1
                        
    val_token_loss = total_token_loss_sum / max(1, total_active_tokens)
    val_sequence_loss = total_seq_loss_sum / max(1, total_sequences)
    val_ppl = math.exp(val_token_loss) if val_token_loss < 100 else float("inf")
    val_bpt = val_token_loss / math.log(2)
    
    # Calculate perplexity per length bucket
    ppl_by_length = {}
    for b_name, b_info in buckets.items():
        if b_info["token_count"] > 0:
            b_loss = b_info["loss_sum"] / b_info["token_count"]
            b_ppl = math.exp(b_loss) if b_loss < 100 else float("inf")
        else:
            b_ppl = float("nan")
        ppl_by_length[f"val/ppl_{b_name}"] = b_ppl
        
    metrics = {
        "val/token_loss": val_token_loss,
        "val/sequence_loss": val_sequence_loss,
        "val/ppl": val_ppl,
        "val/bpt": val_bpt,
        "val/n_tokens": total_active_tokens,
        "val/n_sequences": total_sequences,
        **ppl_by_length
    }
    
    return metrics


def generate_qualitative_examples(
    model: Any,
    tokenizer: Any,
    gen_cfg: Dict[str, Any]
) -> List[Dict[str, str]]:
    """
    Generates text continuations for test prompts for qualitative inspection.
    """
    prompts = gen_cfg.get("prompts", [])
    max_new_tokens = gen_cfg.get("max_new_tokens", 50)
    temperature = gen_cfg.get("temperature", 0.7)
    top_p = gen_cfg.get("top_p", 0.9)
    do_sample = gen_cfg.get("do_sample", True)
    
    device = next(model.parameters()).device
    examples = []
    
    logger.info(f"Generating qualitative examples for {len(prompts)} prompts...")
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        examples.append({
            "prompt": prompt,
            "generated_output": generated_text
        })
        
    return examples


def find_all_checkpoints(checkpoint_dir: str) -> List[str]:
    """
    Finds all checkpoint subdirectories in checkpoint_dir.
    """
    subdirs = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    subdirs = [d for d in subdirs if os.path.isdir(d)]
    
    final_model = os.path.join(checkpoint_dir, "final_lora_model")
    if os.path.isdir(final_model):
        subdirs.append(final_model)
        
    # Sort checkspoints numerically
    def get_step(path):
        name = os.path.basename(path)
        if name.startswith("checkpoint-"):
            try:
                return int(name.split("-")[1])
            except ValueError:
                return 999999
        return 9999999
        
    subdirs.sort(key=get_step)
    return subdirs


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned SLM checkpoints.")
    parser.add_argument(
        "--checkpoint_dir", 
        type=str, 
        default="outputs/pythia-410m-lora", 
        help="Directory containing saved checkpoints or model adapters."
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/eval_config.yaml", 
        help="Path to evaluation configuration file."
    )
    parser.add_argument(
        "--dataset_config", 
        type=str, 
        default="configs/dataset_config.yaml", 
        help="Path to dataset configuration file."
    )
    parser.add_argument(
        "--lora_config", 
        type=str, 
        default="configs/lora_config.yaml", 
        help="Path to LoRA configuration file."
    )
    parser.add_argument(
        "--train_config", 
        type=str, 
        default="configs/train_config.yaml", 
        help="Path to training configuration file."
    )
    args = parser.parse_args()
    
    # 1. Load Configurations
    logger.info(f"Loading eval configuration from {args.config}")
    with open(args.config, "r") as f:
        eval_cfg = yaml.safe_load(f)
        
    logger.info(f"Loading train configuration from {args.train_config}")
    with open(args.train_config, "r") as f:
        train_cfg = yaml.safe_load(f)
        
    model_name = train_cfg["model"]["model_name_or_path"]
    
    logger.info(f"Loading dataset configuration from {args.dataset_config}")
    with open(args.dataset_config, "r") as f:
        dataset_cfg = yaml.safe_load(f)
        
    seq_len = dataset_cfg["preprocessing"]["max_seq_length"]
    
    # 2. Setup Data
    logger.info("Loading validation datasets (streaming mode for fast eval load)...")
    datasets_dict = load_all_datasets(args.dataset_config, streaming=True)
    
    logger.info("Tokenizing and packing validation dataset split...")
    tokenizer_dummy = AutoTokenizer.from_pretrained(model_name)
    val_dataset = tokenize_and_pack(datasets_dict["validation"], tokenizer=tokenizer_dummy, seq_len=seq_len)
    
    # 3. Find Checkpoints
    checkpoints = find_all_checkpoints(args.checkpoint_dir)
    if not checkpoints:
        logger.error(f"No valid checkpoints found in directory: {args.checkpoint_dir}")
        sys.exit(1)
        
    logger.info(f"Found {len(checkpoints)} checkpoints to evaluate: {[os.path.basename(c) for c in checkpoints]}")
    
    # 4. Configure W&B
    log_cfg = eval_cfg.get("logging", {})
    wandb_cfg = log_cfg.get("wandb", {})
    os.environ["WANDB_PROJECT"] = wandb_cfg.get("project", "slm-finetune")
    os.environ["WANDB_NAME"] = wandb_cfg.get("name", "pythia-410m-qlora-evaluation")
    os.environ["WANDB_MODE"] = "offline"  # Offline fallback for reliable local execution
    
    run = wandb.init(project=os.environ["WANDB_PROJECT"], name=os.environ["WANDB_NAME"])
    
    rankings = []
    latest_examples = []
    
    # 5. Evaluate Each Checkpoint
    for ckpt_path in checkpoints:
        ckpt_name = os.path.basename(ckpt_path)
        logger.info(f"\n================ Evaluating Checkpoint: {ckpt_name} ================")
        
        model, tokenizer = load_model_and_tokenizer(ckpt_path, model_name, args.lora_config)
        
        # Compute metrics
        metrics = evaluate_checkpoint(
            model=model,
            eval_dataset=val_dataset,
            batch_size=eval_cfg["eval"].get("batch_size", 4),
            max_samples=eval_cfg["eval"].get("max_eval_samples", 500),
            length_buckets_cfg=eval_cfg["eval"].get("length_buckets", [])
        )
        
        # Qualitative generation on latest checkpoint
        examples = generate_qualitative_examples(model, tokenizer, eval_cfg.get("generation", {}))
        latest_examples = examples
        
        rankings.append({
            "checkpoint": ckpt_name,
            "path": ckpt_path,
            "val_ppl": metrics["val/ppl"],
            "val_token_loss": metrics["val/token_loss"],
            "val_bpt": metrics["val/bpt"]
        })
        
        # Calculate delta_ppl relative to initial step 100 benchmark (PPL 22.51)
        initial_ppl = 22.51
        delta_ppl = initial_ppl - metrics["val/ppl"]
        metrics["val/delta_ppl"] = delta_ppl

        # Log metrics to W&B
        wandb.log({"checkpoint": ckpt_name, **metrics})

        # Log length bucket bar chart to W&B
        bucket_data = [
            [b_name.replace("val/ppl_", ""), b_val] 
            for b_name, b_val in metrics.items() 
            if b_name.startswith("val/ppl_") and not math.isnan(b_val)
        ]
        if bucket_data:
            bucket_table = wandb.Table(data=bucket_data, columns=["length_bucket", "perplexity"])
            wandb.log({
                "val/ppl_by_length_bucket": wandb.plot.bar(
                    bucket_table, "length_bucket", "perplexity", title=f"Perplexity by Length Bucket ({ckpt_name})"
                )
            })
        
        logger.info(f"Checkpoint '{ckpt_name}' - Token Loss: {metrics['val/token_loss']:.4f} | Perplexity: {metrics['val/ppl']:.2f} | BPT: {metrics['val/bpt']:.4f} | Delta PPL: -{delta_ppl:.2f}")
        
        # Clear CUDA cache between checkpoint evaluations
        del model
        torch.cuda.empty_cache()
        
    # 6. Rank Checkpoints
    rankings_df = pd.DataFrame(rankings)
    rankings_df = rankings_df.sort_values(by="val_ppl", ascending=True).reset_index(drop=True)
    rankings_df["rank"] = rankings_df.index + 1
    
    logger.info("\n================ Checkpoint Rankings (by Val PPL) ================")
    logger.info("\n" + rankings_df[["rank", "checkpoint", "val_ppl", "val_token_loss", "val_bpt"]].to_string(index=False))
    
    # Log Rankings Table to W&B
    wandb_table = wandb.Table(dataframe=rankings_df[["rank", "checkpoint", "val_ppl", "val_token_loss", "val_bpt"]])
    wandb.log({"val/checkpoint_rankings": wandb_table})
    
    # Log Qualitative Examples Table to W&B
    if latest_examples:
        ex_df = pd.DataFrame(latest_examples)
        logger.info("\n================ Qualitative Output Examples ================")
        for idx, ex in enumerate(latest_examples, 1):
            logger.info(f"\n--- Example {idx} ---")
            logger.info(f"Prompt:\n{ex['prompt']}")
            logger.info(f"Generated Output:\n{ex['generated_output']}")
            
        ex_table = wandb.Table(dataframe=ex_df)
        wandb.log({"val/examples": ex_table})
        
    logger.info("\nEvaluation complete! All metrics, length bucket charts, and rankings logged to W&B.")
    wandb.finish()


if __name__ == "__main__":
    main()
