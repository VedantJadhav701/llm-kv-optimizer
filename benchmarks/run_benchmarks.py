import sys
import os
from pathlib import Path

# Add project root to sys.path for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pandas as pd
import yaml
from engine.model_loader import ModelLoader
from engine.inference import InferenceEngine

def load_config(config_path: str = "configs/pipeline.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_benchmarks():
    cfg = load_config()
    loader = ModelLoader()
    
    prompts = [
        "What is the principle of KV cache quantization?",
        "Explain the Johnson-Lindenstrauss lemma for vector compression.",
        "How does PolarQuant optimize memory in long-context models?"
    ]
    
    # Support isolated runs via environment variable
    env_method = os.environ.get("METHOD", "").lower()
    if env_method in ["qjl", "polar", "fp16"]:
        methods = [env_method]
        print(f"Running ISOLATED benchmark for: {env_method.upper()}")
    else:
        # Default behavior: run all if enough VRAM (risky on 4GB)
        methods = ["qjl", "polar", "fp16"]

    results = []

    for method in methods:
        print(f"\nBenchmarking Method: {method.upper()}")
        
        # Load model fresh for each method to ensure no memory leak/fragmentation
        model, tokenizer = loader.load_model_and_tokenizer()
        engine = InferenceEngine(model, tokenizer, cfg)
        
        for i, prompt in enumerate(prompts):
            print(f"  Test {i+1}/{len(prompts)}...")
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            
            try:
                # Reduced to 15 tokens to absolutely guarantee stability on 4GB Card
                output = engine.generate(prompt, max_new_tokens=15, method=method)
                
                peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
                
                results.append({
                    "method": method,
                    "prompt_id": i,
                    "latency_sec": output["latency"],
                    "tokens_per_sec": output["tokens_sec"],
                    "kv_memory_mb": output["memory_mb"],
                    "peak_vram_mb": peak_vram,
                    "status": "success"
                })
            except Exception as e:
                print(f"    ERROR in {method}: {e}")
                results.append({
                    "method": method,
                    "prompt_id": i,
                    "status": "oom" if "out of memory" in str(e).lower() else "failed"
                })

        # CRITICAL: Nuclear cleanup between methods
        del engine
        del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import time
        time.sleep(2)

    # Save to CSV
    df = pd.DataFrame(results)
    os.makedirs("experiments", exist_ok=True)
    results_path = "experiments/results.csv"
    df.to_csv(results_path, index=False)
    print(f"\nBenchmarking complete. Results saved to {results_path}")
    
    # Summary
    summary = df.groupby("method").mean(numeric_only=True)
    print("\nBenchmark Summary (Averages):")
    print(summary)

if __name__ == "__main__":
    run_benchmarks()
