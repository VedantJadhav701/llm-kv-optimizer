import torch
import yaml
from engine.model_loader import ModelLoader
from engine.inference import InferenceEngine

def evaluate_improvement():
    loader = ModelLoader()
    with open("configs/pipeline.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    test_queries = [
        "What is the core idea behind the Johnson-Lindenstrauss lemma in the context of LLM KV cache?",
        "How does PolarQuant reduce memory usage compared to standard FP16 KV cache?",
        "Explain the benefit of 4-bit quantization for training on an RTX 3050."
    ]

    print("\n" + "="*50)
    print("PHASE 1: EVALUATING BASE MODEL (WITHOUT LORA)")
    print("="*50)
    # Load base model by explicitly setting lora_path to None
    base_model, tokenizer = loader.load_model_and_tokenizer(lora_path=None)
    base_engine = InferenceEngine(base_model, tokenizer, cfg)
    
    base_responses = []
    for q in test_queries:
        print(f"\nQuery: {q}")
        res = base_engine.generate(q, max_new_tokens=150, method="fp16")
        print(f"Base Response: {res['text']}")
        base_responses.append(res['text'])

    # Clear memory more aggressively for 4GB GPU
    del base_engine
    del base_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2) # Give some time for VRAM to settle

    print("\n" + "="*50)
    print("PHASE 2: EVALUATING FINE-TUNED MODEL (WITH LORA)")
    print("="*50)
    # Load fine-tuned model (default pulls from experiments/)
    ft_model, tokenizer = loader.load_model_and_tokenizer()
    ft_engine = InferenceEngine(ft_model, tokenizer, cfg)

    for i, q in enumerate(test_queries):
        print(f"\nQuery: {q}")
        res = ft_engine.generate(q, max_new_tokens=150, method="fp16")
        print(f"Fine-tuned Response: {res['text']}")
        
        print("-" * 30)
        print(f"Delta Analysis: The fine-tuned model's response is compared against the base for domain specificity.")
    
    # Cleanup FT model
    del ft_engine
    del ft_model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    evaluate_improvement()
