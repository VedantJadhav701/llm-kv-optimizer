import streamlit as st
import torch
import pandas as pd
import plotly.express as px
from engine.model_loader import ModelLoader
from engine.inference import InferenceEngine
import yaml
import time
import os

# --- Page Config ---
st.set_page_config(
    page_title="LLM KV Optimizer",
    page_icon="🧠",
    layout="wide"
)

# --- Memory Reclamation ---
import gc
def clear_vram():
    gc.collect()
    torch.cuda.empty_cache()

# --- Resource Caching ---
@st.cache_resource
def load_engine():
    """
    Cached model loading. This ensures we don't reload the 
    multi-GB model every time the UI updates.
    """
    with open("configs/pipeline.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    loader = ModelLoader()
    model, tokenizer = loader.load_model_and_tokenizer()
    return InferenceEngine(model, tokenizer, cfg)

# --- UI Styles ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1a1c24;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_value=True)

# --- Sidebar ---
st.sidebar.title("⚙️ Optimization Settings")
method = st.sidebar.selectbox(
    "KV Cache Method",
    options=["fp16", "qjl", "polar"],
    index=1,
    help="FP16: Standard | QJL: 23x Savings | Polar: 3x Savings"
)

max_tokens = st.sidebar.slider("Max New Tokens", 10, 100, 40)
temperature = st.sidebar.slider("Creativity (Temp)", 0.1, 1.0, 0.7)

st.sidebar.markdown("---")
st.sidebar.info(f"**Hardware**: RTX 3050 (4GB)\n\n**Method**: {method.upper()}")

if st.sidebar.button("Clear VRAM Cache"):
    clear_vram()
    st.sidebar.success("VRAM Cleared!")

# --- Main UI ---
st.title("🧠 LLM KV Optimizer Dashboard")
st.markdown("### Interactive Inference & Real-time Memory Visualization")

query = st.text_area("Enter your prompt:", "Explain the concept of KV cache quantization in one sentence.", height=100)

if st.button("🚀 Run Optimized Inference"):
    with st.spinner(f"Generating with {method.upper()} optimization..."):
        try:
            engine = load_engine()
            
            # Reset peak stats
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            # Run generation
            result = engine.generate(query, max_new_tokens=max_tokens, method=method)
            
            # Display Results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📝 Model Response")
                st.write(result['text'])
            
            with col2:
                st.subheader("📊 Metrics")
                st.metric("Tokens/sec", f"{result['tokens_sec']:.2f}")
                st.metric("KV Memory", f"{result['memory_mb']:.2f} MB")
                
                # Baseline Comparison Data
                # These are averages from our recent benchmarks
                baseline_kv = 1695.9
                current_kv = result['memory_mb']
                reduction = baseline_kv / current_kv if current_kv > 0 else 1
                
                st.metric("Memory Reduction", f"{reduction:.1f}x", delta=f"{reduction-1:.1f}x", delta_color="normal")

            # Visualization
            st.markdown("---")
            st.subheader("📈 Memory Comparison (Ideal vs Actual)")
            
            plot_data = pd.DataFrame({
                "Method": ["Baseline (FP16)", f"Current ({method.upper()})"],
                "KV Memory (MB)": [baseline_kv, current_kv]
            })
            
            fig = px.bar(
                plot_data, 
                x="Method", 
                y="KV Memory (MB)",
                color="Method",
                color_discrete_map={"Baseline (FP16)": "#ff4b4b", f"Current ({method.upper()})": "#00d26a"},
                title="KV Cache Memory Usage (lower is better)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error during inference: {e}")
            if "out of memory" in str(e).lower():
                st.warning("⚠️ OOM detected. Clearing VRAM. Try reducing tokens.")
                clear_vram()

# --- Footer ---
st.markdown("---")
st.caption("LLM-KV-Optimizer | Optimized for RTX 3050 | Built with Streamlit")
