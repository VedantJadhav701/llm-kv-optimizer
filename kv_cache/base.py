from abc import ABC, abstractmethod
import torch
from typing import Tuple, Optional

class KVCache(ABC):
    """Base class for pluggable KV cache optimization modules."""
    
    @abstractmethod
    def update(self, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with new key and value tensors.
        Returns the (possibly quantized/transformed) full cache.
        """
        pass

    @abstractmethod
    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the full cached keys and values."""
        pass
        
    @abstractmethod
    def clear(self):
        """Clears the cache state."""
        pass

    @property
    @abstractmethod
    def memory_usage(self) -> float:
        """Returns memory usage in MB."""
        pass
