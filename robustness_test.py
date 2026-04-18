import torch
import gc
import yaml
import time
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from engine.model_loader import ModelLoader
from engine.inference import InferenceEngine

def clear_memory():
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass # Ignore sticky OOM during cleanup

def run_robustness_check():
    print("🚀 Starting Final Robustness Test Suite (4GB Optimized)...")
    
    # Load Config
    with open("configs/pipeline.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # 1. Load Model (Once for efficiency)
    loader = ModelLoader()
    model, tokenizer = loader.load_model_and_tokenizer()
    engine = InferenceEngine(model, tokenizer, cfg)

    test_queries = [
        {
            "category": "TECHNICAL",
            "prompt": "How does the QJL projection matrix reduce KV cache memory usage?",
            "method": "qjl"
        },
        {
            "category": "REASONING",
            "prompt": "If I have 4GB VRAM and a 0.5B model takes 500MB, calculate how many tokens I can fit if each token takes 1MB in FP16.",
            "method": "polar"
        },
        {
            "category": "CREATIVE",
            "prompt": "Write a short 4-line poem about an LLM living in an RTX 3050.",
            "method": "qjl"
        },
        {
            "category": "COMPARISON",
            "prompt": "What is the primary difference between Cartesian and Polar coordinate quantization in KV caches?",
            "method": "polar"
        }
    ]

    print("\n" + "="*50)
    print("STABILITY RUN: COMMENCING QUERIES")
    print("="*50 + "\n")

    results_log = "experiments/final_robustness_log.txt"
    os.makedirs("experiments", exist_ok=True)

    with open(results_log, "w", encoding="utf-8") as f:
        f.write("=== FINAL LLM-KV-OPTIMIZER ROBUSTNESS LOG ===\n\n")

        for i, task in enumerate(test_queries):
            print(f"[{i+1}/{len(test_queries)}] Testing {task['category']} (Method: {task['method'].upper()})...")
            
            clear_memory()
            try:
                # Use a moderate token count for robustness
                start = time.time()
                result = engine.generate(task['prompt'], max_new_tokens=35, method=task['method'])
                duration = time.time() - start

                output_str = (
                    f"QUERY: {task['prompt']}\n"
                    f"METHOD: {task['method'].upper()}\n"
                    f"RESPONSE: {result['text']}\n"
                    f"STATS: {result['tokens_sec']:.2f} tok/s | {result['memory_mb']:.2f} MB KV\n"
                    f"--------------------------------------------------\n"
                )
                
                print(f"  ✅ Success! ({result['tokens_sec']:.2f} tok/s)")
                f.write(output_str)

            except Exception as e:
                err_msg = f"  ❌ FAILED: {str(e)}\n"
                print(err_msg)
                f.write(f"QUERY: {task['prompt']}\nERROR: {str(e)}\n" + "-"*50 + "\n")
            
            # Brief pause to let hardware cool down/clear
            time.sleep(1)

    print("\n" + "="*50)
    print(f"TESTING COMPLETE. Results saved to: {results_log}")
    print("="*50)

if __name__ == "__main__":
    run_robustness_check()
