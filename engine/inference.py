import torch
import time
import yaml
from typing import List, Dict, Tuple, Optional
from kv_cache.base import KVCache
from kv_cache.fp16 import FP16Cache
from kv_cache.qjl import QJLCache
from kv_cache.polar import PolarQuantCache
from transformers import DynamicCache

class InferenceEngine:
    """Handles text generation with pluggable KV cache optimizations."""
    
    def __init__(self, model, tokenizer, config: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = model.device
        self.num_layers = model.config.num_hidden_layers

    def _get_cache_modules(self, method: str) -> List[KVCache]:
        caches = []
        for _ in range(self.num_layers):
            if method == "fp16":
                caches.append(FP16Cache())
            elif method == "qjl":
                d_model = self.model.config.hidden_size // self.model.config.num_attention_heads
                caches.append(QJLCache(d_model=d_model))
            elif method == "polar":
                caches.append(PolarQuantCache())
            else:
                raise ValueError(f"Unknown KV cache method: {method}")
        return caches

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 50, method: str = "fp16") -> Dict:
        """
        Custom generation loop with aggressive memory reclamation for 4GB GPUs.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        
        # Initialize parallel cache modules for each layer
        layer_caches = self._get_cache_modules(method)
        
        start_time = time.time()
        generated_ids = input_ids
        
        # current_past_key_values will store the decompressed/usable state for the next step
        current_past_key_values = None
        
        # Import cleanup tools
        import gc
        
        for step in range(max_new_tokens):
            # Standard generation step
            model_inputs = generated_ids[:, -1:] if current_past_key_values is not None else generated_ids
            
            outputs = self.model(
                input_ids=model_inputs,
                past_key_values=current_past_key_values,
                use_cache=True
            )
            
            logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Step: Extract, Optimize and Update custom caches
            raw_past_key_values = outputs.past_key_values
            
            # Initialize a proper DynamicCache object for the model to consume
            cache_obj = DynamicCache()
            if raw_past_key_values is not None:
                for i, (k, v) in enumerate(raw_past_key_values):
                    # Update and compress; new_k, new_v are FULL reconstructed KV
                    new_k, new_v = layer_caches[i].update(k, v)
                    cache_obj.update(new_k, new_v, i)
                
                current_past_key_values = cache_obj
            else:
                current_past_key_values = None
            
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            
            # --- NUCLEAR VRAM RECLAMATION ---
            # Delete massive temporary objects immediately
            del outputs
            del logits
            
            # Periodic aggressive cleanup (every step to ensure no fragmentation)
            gc.collect()
            torch.cuda.empty_cache()
            
            if next_token_id == self.tokenizer.eos_token_id:
                break
                
        latency = time.time() - start_time
        num_tokens = generated_ids.shape[1] - input_ids.shape[1]
        
        # Calculate aggregate memory usage across all layers
        total_memory_mb = sum(c.memory_usage for c in layer_caches)
        
        return {
            "text": self.tokenizer.decode(generated_ids[0], skip_special_tokens=True),
            "tokens_sec": num_tokens / latency if latency > 0 else 0,
            "memory_mb": total_memory_mb,
            "latency": latency
        }

if __name__ == "__main__":
    import sys
    import os
    # Add project root to sys.path for absolute imports
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    
    from engine.model_loader import ModelLoader
    loader = ModelLoader()
    m, t = loader.load_model_and_tokenizer()
    with open("configs/pipeline.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    engine = InferenceEngine(m, t, cfg)
    res = engine.generate("Explain KV cache optimization.", method="qjl")
    print(res)
