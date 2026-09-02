import os
import sys
import time
import argparse
import logging
import json
import yaml
import torch
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("inference")


def load_model_and_tokenizer(
    checkpoint_dir: str,
    model_name: str = "EleutherAI/pythia-410m",
    lora_cfg_path: str = "configs/lora_config.yaml"
) -> Tuple[Any, Any]:
    """
    Loads base model in 4-bit precision and attaches the LoRA adapter.
    """
    logger.info(f"Loading LoRA/Quantization config from {lora_cfg_path}")
    if os.path.exists(lora_cfg_path):
        with open(lora_cfg_path, "r") as f:
            lora_cfg = yaml.safe_load(f)
        bnb_cfg = lora_cfg.get("quantization", {})
        compute_dtype = getattr(torch, bnb_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
        
        quant_config = BitsAndBytesConfig(
            load_in_4bit=bnb_cfg.get("load_in_4bit", True),
            bnb_4bit_quant_type=bnb_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=bnb_cfg.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=compute_dtype
        )
    else:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    logger.info(f"Loading 4-bit base model '{model_name}'...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto"
    )

    if checkpoint_dir and os.path.exists(checkpoint_dir):
        abs_ckpt = os.path.abspath(checkpoint_dir)
        logger.info(f"Attaching LoRA adapter checkpoint from: {abs_ckpt}")
        model = PeftModel.from_pretrained(base_model, abs_ckpt)
    else:
        logger.warning(f"Checkpoint '{checkpoint_dir}' not found or empty. Using base model.")
        model = base_model

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_response(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> str:
    """
    Generates text response for a single prompt.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def benchmark_inference(
    model: Any,
    tokenizer: Any,
    prompt: str,
    gen_lengths: List[int],
    num_runs: int = 10
) -> List[Dict[str, Any]]:
    """
    Benchmarks latency (ms/token), throughput (tokens/sec), and peak VRAM usage.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_token_count = inputs["input_ids"].shape[1]

    results = []

    logger.info(f"Beginning benchmark over target generation lengths: {gen_lengths} (trials={num_runs})...")

    # Warmup runs
    logger.info("Executing warmup runs...")
    with torch.no_grad():
        for _ in range(2):
            model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    for gen_len in gen_lengths:
        logger.info(f"\nBenchmarking target max_new_tokens={gen_len}...")
        latencies_ms_per_token = []
        throughputs_tokens_per_sec = []
        vram_usages_mb = []

        for run_idx in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=gen_len,
                    min_new_tokens=gen_len,  # enforce uniform generation length for accurate token timing
                    do_sample=False,         # deterministic greedy decoding for benchmarking consistency
                    pad_token_id=tokenizer.eos_token_id
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.perf_counter()

            total_gen_time_sec = end_time - start_time
            generated_tokens = outputs[0].shape[0] - prompt_token_count

            if generated_tokens <= 0:
                continue

            ms_per_token = (total_gen_time_sec / generated_tokens) * 1000.0
            tokens_per_sec = generated_tokens / total_gen_time_sec
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

            latencies_ms_per_token.append(ms_per_token)
            throughputs_tokens_per_sec.append(tokens_per_sec)
            vram_usages_mb.append(peak_vram_mb)

        avg_latency = float(pd.Series(latencies_ms_per_token).mean())
        avg_throughput = float(pd.Series(throughputs_tokens_per_sec).mean())
        avg_vram = float(pd.Series(vram_usages_mb).mean())

        logger.info(f"Target Length: {gen_len} | Avg Latency: {avg_latency:.2f} ms/token | Throughput: {avg_throughput:.2f} tok/s | Peak VRAM: {avg_vram:.2f} MB")

        results.append({
            "gen_length": gen_len,
            "avg_latency_ms_per_token": round(avg_latency, 2),
            "avg_throughput_tokens_per_sec": round(avg_throughput, 2),
            "peak_vram_mb": round(avg_vram, 2),
            "num_runs": num_runs,
            "prompt_length_tokens": prompt_token_count
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="SLM Fine-tuning Inference & Benchmarking Pipeline")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="outputs/pythia-410m-lora/final_lora_model",
        help="Path to saved LoRA adapter checkpoint."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="EleutherAI/pythia-410m",
        help="Base model HF hub identifier."
    )
    parser.add_argument(
        "--lora_config",
        type=str,
        default="configs/lora_config.yaml",
        help="Path to LoRA configuration file."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time in a small village,",
        help="Prompt for single generation or benchmark input."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Max new tokens to generate for single prompt mode."
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run latency, throughput, and VRAM benchmarking."
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help="Number of benchmark trials per generation length."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs/benchmarks",
        help="Directory to save benchmark reports."
    )
    args = parser.parse_args()

    # 1. Load Model & Tokenizer
    model, tokenizer = load_model_and_tokenizer(
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model_name,
        lora_cfg_path=args.lora_config
    )

    # 2. Benchmark Mode or Interactive Generation
    if args.benchmark:
        gen_lengths = [50, 100, 200]
        benchmark_results = benchmark_inference(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            gen_lengths=gen_lengths,
            num_runs=args.num_runs
        )

        # Print Summary Table
        df = pd.DataFrame(benchmark_results)
        logger.info("\n================ Benchmark Summary Table ================")
        logger.info("\n" + df.to_string(index=False))

        # Save to Output Directory
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        json_path = os.path.join(args.output_dir, f"benchmark_report_{timestamp}.json")
        csv_path = os.path.join(args.output_dir, f"benchmark_report_{timestamp}.csv")

        with open(json_path, "w") as f:
            json.dump(benchmark_results, f, indent=2)

        df.to_csv(csv_path, index=False)
        logger.info(f"\nSaved benchmark results to:\n - {json_path}\n - {csv_path}")

    else:
        logger.info("\nGenerating single response...")
        output_text = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens
        )
        logger.info(f"\nPrompt:\n{args.prompt}")
        logger.info(f"\nGenerated Response:\n{output_text}")


if __name__ == "__main__":
    main()
