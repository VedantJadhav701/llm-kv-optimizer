from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import yaml
import sys
import os
# Add project root to sys.path for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.model_loader import ModelLoader
from engine.inference import InferenceEngine
from typing import Optional

app = FastAPI(title="LLM KV Optimizer API")

# Global state for model and engine
model = None
tokenizer = None
engine = None
config = None

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 100
    method: Optional[str] = "fp16" # fp16, qjl, polar

class GenerateResponse(BaseModel):
    text: str
    latency: float
    memory_usage_mb: float
    tokens_per_sec: float

@app.on_event("startup")
async def startup_event():
    global model, tokenizer, engine, config
    print("Loading model and configuration...")
    loader = ModelLoader()
    model, tokenizer = loader.load_model_and_tokenizer()
    with open("configs/pipeline.yaml", "r") as f:
        config = yaml.safe_load(f)
    engine = InferenceEngine(model, tokenizer, config)
    print("API Ready.")

@app.get("/")
async def root():
    return {"message": "LLM KV Optimizer API is running"}

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        output = engine.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_tokens,
            method=request.method
        )
        return GenerateResponse(
            text=output["text"],
            latency=output["latency"],
            memory_usage_mb=output["memory_mb"],
            tokens_per_sec=output["tokens_sec"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
