import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path
from typing import Optional

class ModelLoader:
    """Loads and configures the LLM for inference or training."""
    
    def __init__(self, config_path: str = "configs/pipeline.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.model_name = self.config["model"]["name"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model_and_tokenizer(self, lora_path: Optional[str] = "experiments/qwen2_lora_adapter"):
        """Loads the tokenizer and model with 4-bit quantization."""
        print(f"Loading tokenizer: {self.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # BitsAndBytes for 4-bit quantization (required for 4GB GPU)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        print(f"Loading model: {self.model_name} on {self.device}")
        
        # Define a hard VRAM budget for the model weights (leave 1.5GB for overhead/cache)
        # This is critical for 4GB GPUs to prevent 'Sticky OOM'
        max_memory = {0: "2.5GiB"} if self.device == "cuda" else None
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            max_memory=max_memory,
            attn_implementation="eager", # 'eager' is more memory-stable than 'sdpa' on small GPUs
            trust_remote_code=True
        )

        if lora_path and Path(lora_path).exists():
            print(f"Applying LoRA weights from: {lora_path}")
            model = PeftModel.from_pretrained(model, lora_path)
        else:
            print("No LoRA weights found, using base model.")
            
        return model, tokenizer

if __name__ == "__main__":
    loader = ModelLoader()
    m, t = loader.load_model_and_tokenizer()
    print("Model loaded successfully.")
