import torch
import yaml
import json
import os
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Resolve OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def load_config(config_path: str = "configs/pipeline.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def format_prompt(example):
    return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""

def train():
    cfg = load_config()
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    # -------------------------
    # Load Dataset
    # -------------------------
    if not os.path.exists(data_cfg["dataset_path"]):
        raise FileNotFoundError(f"Dataset not found at {data_cfg['dataset_path']}. Run data_pipeline/dataset_generator.py first.")

    with open(data_cfg["dataset_path"], "r") as f:
        data = json.load(f)

    dataset = Dataset.from_list(data)

    # -------------------------
    # Tokenizer
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(example):
        full_prompt = format_prompt(example)

        tokenized = tokenizer(
            full_prompt,
            truncation=True,
            max_length=model_cfg["max_length"],
        )

        # Create labels
        labels = tokenized["input_ids"].copy()

        # Mask instruction + input part
        response_marker = "### Response:"
        response_start = full_prompt.find(response_marker)
        
        if response_start == -1:
            # Fallback if marker not found
            tokenized["labels"] = labels
            return tokenized

        # Calculate mask length based on tokens
        instruction_part = full_prompt[:response_start + len(response_marker)]
        instruction_tokens = tokenizer(
            instruction_part,
            truncation=True,
            max_length=model_cfg["max_length"],
        )["input_ids"]
        
        mask_len = len(instruction_tokens)
        
        # Mask everything except response
        # Labels should be -100 for ignored tokens in CrossEntropyLoss
        labels[:mask_len] = [-100] * min(mask_len, len(labels))

        tokenized["labels"] = labels
        return tokenized

    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

    # -------------------------
    # Model (4-bit)
    # -------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model)

    # 🔥 IMPORTANT (memory optimization for 4GB VRAM)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # -------------------------
    # LoRA
    # -------------------------
    lora_config = LoraConfig(
        r=train_cfg["r"],
        lora_alpha=train_cfg["lora_alpha"],
        target_modules=train_cfg["target_modules"],
        lora_dropout=train_cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # -------------------------
    # Training Args
    # -------------------------
    training_args = TrainingArguments(
        output_dir="./experiments/checkpoints",
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"] if "per_device_train_batch_size" in train_cfg else train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=float(train_cfg["learning_rate"]),
        num_train_epochs=train_cfg["epochs"],
        logging_steps=10,
        save_steps=100,
        fp16=True,
        optim="paged_adamw_32bit",
        report_to="none",
        save_total_limit=2,
        remove_unused_columns=False
    )

    # -------------------------
    # Trainer
    # -------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print("🚀 Starting production-grade training (RTX 3050 Optimized)...")
    trainer.train()

    # Save final model
    os.makedirs("./experiments/qwen2_lora_adapter", exist_ok=True)
    model.save_pretrained("./experiments/qwen2_lora_adapter")
    print("✅ Training complete. Adapter saved to ./experiments/qwen2_lora_adapter")

if __name__ == "__main__":
    train()
