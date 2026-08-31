import logging
from typing import Dict, List, Any
from datasets import Dataset

logger = logging.getLogger(__name__)

def tokenize_and_pack(dataset: Dataset, tokenizer: Any, seq_len: int) -> Dataset:
    """
    Tokenizes the input dataset's 'text' column and applies sequence packing.
    Concatenates tokenized examples and chunks them into fixed-length blocks of `seq_len`.
    Also configures the Pythia/GPT-NeoX tokenizer padding/eos settings, adds 'labels' for Causal LM,
    and logs dataset statistics and sanity checks.
    """
    logger.info("Starting tokenization and sequence packing...")
    
    # 1. Align/setup tokenizer pad and eos tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set tokenizer.pad_token to tokenizer.eos_token: {tokenizer.eos_token}")
        
    # 2. Tokenize the 'text' field
    def tokenize_fn(examples: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        # Return input_ids and attention_mask without truncation/padding (letting pack handle size)
        return tokenizer(examples["text"], add_special_tokens=True)
        
    tokenized_ds = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing text sequences"
    )
    
    # 3. Concatenate and pack into chunk sizes of seq_len
    def group_texts(examples: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
        # Concatenate all sequences of tokens and masks
        concatenated_ids = []
        concatenated_masks = []
        
        for ids, mask in zip(examples["input_ids"], examples["attention_mask"]):
            # For causal LM, append the EOS token ID if it is not already at the end
            if len(ids) > 0 and ids[-1] != tokenizer.eos_token_id:
                ids.append(tokenizer.eos_token_id)
                mask.append(1)
            concatenated_ids.extend(ids)
            concatenated_masks.extend(mask)
            
        total_length = len(concatenated_ids)
        # Drop the remainder chunk to ensure exact sequence lengths
        if total_length >= seq_len:
            total_length = (total_length // seq_len) * seq_len
            
        # Split into blocks of seq_len
        input_ids_chunks = [concatenated_ids[i : i + seq_len] for i in range(0, total_length, seq_len)]
        attention_mask_chunks = [concatenated_masks[i : i + seq_len] for i in range(0, total_length, seq_len)]
        
        # Add labels = input_ids for causal language modeling
        labels_chunks = [chunk.copy() for chunk in input_ids_chunks]
        
        return {
            "input_ids": input_ids_chunks,
            "attention_mask": attention_mask_chunks,
            "labels": labels_chunks
        }
        
    packed_ds = tokenized_ds.map(
        group_texts,
        batched=True,
        desc=f"Packing sequences into chunks of {seq_len}"
    )
    
    # Sanity checks and diagnostic logging
    original_size = len(dataset)
    packed_size = len(packed_ds)
    total_tokens = packed_size * seq_len
    
    logger.info(f"Dataset Size Before Packing: {original_size} sequences")
    logger.info(f"Dataset Size After Packing: {packed_size} sequences (fixed length: {seq_len})")
    logger.info(f"Total processed tokens: {total_tokens}")
    
    if packed_size > 0:
        sample_ids = packed_ds[0]["input_ids"]
        decoded_sample = tokenizer.decode(sample_ids[:100])
        logger.info(f"Sanity Check - Decoded start of first sample: {repr(decoded_sample)}")
        
    return packed_ds
