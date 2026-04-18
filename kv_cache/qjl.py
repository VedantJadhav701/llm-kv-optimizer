import torch
import numpy as np
from .base import KVCache
from typing import Tuple, Optional

class QJLCache(KVCache):
    """
    1-bit Quantized Johnson-Lindenstrauss KV Cache.
    Reduces memory by projecting vectors into a lower-dimensional space 
    and preserving distances via random projection.
    """
    
    def __init__(self, d_model: int, compression_ratio: float = 0.5, seed: int = 42):
        self.d_model = d_model
        self.reduced_dim = int(d_model * compression_ratio)
        self.seed = seed
        
        # Initialize random projection matrix
        # We use a fixed seed for consistency across cache updates
        torch.manual_seed(seed)
        self.proj_matrix = torch.randn(d_model, self.reduced_dim) / np.sqrt(self.reduced_dim)
        
        self.q_keys: Optional[torch.Tensor] = None
        self.q_values: Optional[torch.Tensor] = None

    def _quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Projects and applies sign quantization with enforced float16 precision."""
        # Force EVERYTHING to half (float16) for modern LLM compatibility on 4GB GPUs
        proj = self.proj_matrix.to(device=tensor.device, dtype=torch.half)
        input_tensor = tensor.to(torch.half)
        
        # Project: (..., d_model) @ (d_model, reduced_dim) -> (..., reduced_dim)
        projected = input_tensor @ proj
        # Sign quantization (1 bit per element)
        return torch.sign(projected)

    def update(self, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q_k = self._quantize(key)
        q_v = self._quantize(value)
        
        if self.q_keys is None:
            self.q_keys = q_k
            self.q_values = q_v
        else:
            self.q_keys = torch.cat([self.q_keys, q_k], dim=-2)
            self.q_values = torch.cat([self.q_values, q_v], dim=-2)
            
        return self.get()

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.q_keys is None:
            return None, None
            
        # Hard cast projection matrix for reconstruction
        proj_t = self.proj_matrix.to(device=self.q_keys.device, dtype=torch.half).T
        
        reconstructed_k = self.q_keys.to(torch.half) @ proj_t
        reconstructed_v = self.q_values.to(torch.half) @ proj_t
        return reconstructed_k, reconstructed_v

    def clear(self):
        self.q_keys = None
        self.q_values = None

    @property
    def memory_usage(self) -> float:
        if self.q_keys is None:
            return 0.0
        # QJL stores 1-bit values (effectively). 
        # In this PyTorch prototype, we use sign tensors (float), 
        # but in a production C++ kernel, this would be bit-packed.
        # We'll calculate "ideal" 1-bit memory usage for the benchmark.
        num_elements = self.q_keys.numel() + self.q_values.numel()
        return (num_elements / 8) / (1024 * 1024)
