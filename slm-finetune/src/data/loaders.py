import os
import yaml
import logging
from typing import Dict, List, Optional, Any
from datasets import load_dataset, concatenate_datasets, interleave_datasets, Dataset, DatasetDict

logger = logging.getLogger(__name__)

def format_row(example: Dict[str, Any], dataset_cfg: Dict[str, Any], features: Any) -> Dict[str, str]:
    """
    Normalizes a single row/example from a dataset to a unified 'text' format
    based on the format templates specified in the dataset configuration.
    """
    fmt = dataset_cfg.get("format", "raw")
    text_col = dataset_cfg.get("text_column", "text")
    
    # Pre-process labels or nested structures if present
    processed_example = dict(example)
    
    # ClassLabel integer-to-string mapping for templates (e.g. labels in ag_news, yelp)
    if "label" in processed_example and features and "label" in features:
        from datasets import ClassLabel
        label_feature = features["label"]
        if isinstance(label_feature, ClassLabel):
            try:
                processed_example["label"] = label_feature.int2str(example["label"])
            except Exception as e:
                logger.warning(f"Could not convert label int to str: {e}")
                
    if fmt == "raw":
        val = processed_example.get(text_col, "")
        return {"text": str(val) if val is not None else ""}
        
    elif fmt == "template":
        template_str = dataset_cfg.get("template", "")
        
        # Special preprocessing for nested keys such as squad answers
        if "squad" in dataset_cfg.get("name", "").lower():
            ans_text = ""
            if "answers" in processed_example and isinstance(processed_example["answers"], dict):
                texts = processed_example["answers"].get("text", [])
                if len(texts) > 0:
                    ans_text = texts[0]
            processed_example["answers_text"] = ans_text
            # Normalize the template placeholder
            template_str = template_str.replace("{answers[text][0]}", "{answers_text}")
            
        try:
            formatted_text = template_str.format(**processed_example)
            return {"text": formatted_text}
        except KeyError as e:
            logger.error(f"KeyError formatting template for dataset {dataset_cfg.get('name')}: missing {e}")
            return {"text": ""}
            
    elif fmt == "dialog":
        # daily_dialog turns are list of strings in the 'dialog' key
        dialog_turns = processed_example.get("dialog", [])
        if isinstance(dialog_turns, list):
            formatted_text = "\n".join([str(turn) for turn in dialog_turns])
            return {"text": formatted_text}
        return {"text": str(dialog_turns)}
        
    else:
        val = processed_example.get(text_col, "")
        return {"text": str(val) if val is not None else ""}


def load_single_dataset(dataset_cfg: Dict[str, Any], is_val: bool = False) -> Optional[Dataset]:
    """
    Loads and normalizes a single dataset according to its configuration.
    """
    name = dataset_cfg.get("name")
    path = dataset_cfg.get("path")
    config = dataset_cfg.get("config")
    
    # Determine the split to use
    if is_val:
        split = dataset_cfg.get("validation_split")
        if not split:
            # Will be split from train dataset later
            return None
    else:
        split = dataset_cfg.get("train_split", "train")
        
    logger.info(f"Loading dataset: {name} (path: {path}, split: {split})")
    
    try:
        # Load dataset
        if config:
            dataset = load_dataset(path, config, split=split)
        else:
            dataset = load_dataset(path, split=split)
            
        # Apply max_samples capping
        max_samples = dataset_cfg.get("max_samples")
        if max_samples and not is_val:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        elif max_samples and is_val:
            # For validation split, cap it proportionally to keep it light
            val_cap = max(100, int(max_samples * 0.1))
            dataset = dataset.select(range(min(val_cap, len(dataset))))
            
        # Normalize features
        features = dataset.features
        
        # Map/Format to text column
        normalized_dataset = dataset.map(
            lambda x: format_row(x, dataset_cfg, features),
            remove_columns=dataset.column_names,
            desc=f"Normalizing {name} to unified 'text' format"
        )
        return normalized_dataset
        
    except Exception as e:
        logger.error(f"Failed to load dataset {name}: {e}", exc_info=True)
        return None


def load_all_datasets(config_path: str) -> DatasetDict:
    """
    Loads all datasets listed in the configuration file, normalizes them,
    and returns a combined DatasetDict containing 'train' and 'validation' splits.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    dataset_configs = config.get("datasets", [])
    train_datasets = []
    val_datasets = []
    
    for cfg in dataset_configs:
        name = cfg.get("name")
        # Load Train split
        train_ds = load_single_dataset(cfg, is_val=False)
        if train_ds is None:
            logger.warning(f"Skipping dataset {name} as it failed to load.")
            continue
            
        # Load Validation split or create one
        val_split_name = cfg.get("validation_split")
        if val_split_name:
            val_ds = load_single_dataset(cfg, is_val=True)
            if val_ds is not None:
                val_datasets.append(val_ds)
                train_datasets.append(train_ds)
            else:
                logger.warning(f"Could not load specified val split for {name}, fallback to splitting train.")
                val_split_name = None
                
        if not val_split_name:
            # Need to split train
            val_pct = cfg.get("val_split_percentage", 5) / 100.0
            if len(train_ds) > 10:
                split_ds = train_ds.train_test_split(test_size=val_pct, seed=42)
                train_datasets.append(split_ds["train"])
                val_datasets.append(split_ds["test"])
            else:
                train_datasets.append(train_ds)
                
    if not train_datasets:
        raise ValueError("No datasets were successfully loaded.")
        
    # Interleave or Concatenate datasets
    # Since we want a unified mixed corpus, we can use interleave_datasets
    # If any errors occur or no probabilities are set, fallback to concatenation
    logger.info("Combining all normalized datasets...")
    try:
        combined_train = interleave_datasets(train_datasets, stopping_strategy="all_exhausted")
        combined_val = interleave_datasets(val_datasets, stopping_strategy="all_exhausted")
    except Exception as e:
        logger.warning(f"Could not interleave datasets: {e}. Falling back to concatenation.")
        combined_train = concatenate_datasets(train_datasets)
        combined_val = concatenate_datasets(val_datasets)
        
    return DatasetDict({
        "train": combined_train,
        "validation": combined_val
    })
