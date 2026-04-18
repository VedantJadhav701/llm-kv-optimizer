import torch
from .base import KVCache
from typing import Tuple, Optional

class FP16Cache(KVCache):
    """Baseline FP16 KV cache (standard HuggingFace-like behavior)."""
    
    def __init__(self):
        self.keys: Optional[torch.Tensor] = None
        self.values: Optional[torch.Tensor] = None

    def update(self, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.keys is None:
            self.keys = key
            self.values = value
        else:
            self.keys = torch.cat([self.keys, key], dim=-2)
            self.values = torch.cat([self.values, value], dim=-2)
        return self.keys, self.values

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.keys, self.values

    def clear(self):
        self.keys = None
        self.values = None

    @property
    def memory_usage(self) -> float:
        if self.keys is None:
            return 0.0
        # memory in MB: (num_elements * bytes_per_element) / (1024 * 1024)
        k_mem = (self.keys.numel() * self.keys.element_size()) / (1024 * 1024)
        v_mem = (self.values.numel() * self.values.element_size()) / (1024 * 1024)
        return k_mem + v_mem
