import torch
from .base import KVCache
from typing import Tuple, Optional

class PolarQuantCache(KVCache):
    """
    PolarQuant KV Cache.
    Converts Cartesion (x,y) pairs to Polar (r, phi) coordinates.
    Magnitudes (r) are kept in higher precision, while angles (phi) are quantized.
    """
    
    def __init__(self, bits_phi: int = 4):
        self.bits_phi = bits_phi
        self.num_bins = 2 ** bits_phi
        self.r_keys: Optional[torch.Tensor] = None
        self.phi_keys: Optional[torch.Tensor] = None
        self.r_vals: Optional[torch.Tensor] = None
        self.phi_vals: Optional[torch.Tensor] = None

    def _cartesian_to_polar(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Converts last dim pairs to r and phi."""
        shape = tensor.shape
        # Reshape to (..., d/2, 2)
        paired = tensor.view(*shape[:-1], -1, 2)
        x = paired[..., 0]
        y = paired[..., 1]
        
        r = torch.sqrt(x**2 + y**2)
        phi = torch.atan2(y, x)
        return r, phi

    def _polar_to_cartesian(self, r: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Converts r and phi back to (x, y) coordinates."""
        x = r * torch.cos(phi)
        y = r * torch.sin(phi)
        
        # Stack and reshape back
        combined = torch.stack([x, y], dim=-1)
        return combined.view(*combined.shape[:-2], -1)

    def _quantize_phi(self, phi: torch.Tensor) -> torch.Tensor:
        """Quantizes the angle into discrete bins."""
        # Normalize phi from [-pi, pi] to [0, 1]
        normalized = (phi + torch.pi) / (2 * torch.pi)
        # Quantize to bins
        quantized = torch.round(normalized * (self.num_bins - 1)) / (self.num_bins - 1)
        # Rescale back to [-pi, pi]
        return (quantized * 2 * torch.pi) - torch.pi

    def update(self, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rk, phik = self._cartesian_to_polar(key)
        rv, phiv = self._cartesian_to_polar(value)
        
        # Quantize phi
        phik_q = self._quantize_phi(phik)
        phiv_q = self._quantize_phi(phiv)
        
        if self.r_keys is None:
            self.r_keys, self.phi_keys = rk, phik_q
            self.r_vals, self.phi_vals = rv, phiv_q
        else:
            self.r_keys = torch.cat([self.r_keys, rk], dim=-2)
            self.phi_keys = torch.cat([self.phi_keys, phik_q], dim=-2)
            self.r_vals = torch.cat([self.r_vals, rv], dim=-2)
            self.phi_vals = torch.cat([self.phi_vals, phiv_q], dim=-2)
            
        return self.get()

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.r_keys is None:
            return None, None
            
        k = self._polar_to_cartesian(self.r_keys, self.phi_keys)
        v = self._polar_to_cartesian(self.r_vals, self.phi_vals)
        return k, v

    def clear(self):
        self.r_keys = None
        self.phi_keys = None
        self.r_vals = None
        self.phi_vals = None

    @property
    def memory_usage(self) -> float:
        if self.r_keys is None:
            return 0.0
        # Estimation: r is 16-bit (2 bytes), phi is 4-bit (0.5 bytes)
        # Number of pairs = num_elements / 2
        num_pairs = self.r_keys.numel() + self.r_vals.numel()
        mem_bytes = num_pairs * 2.5 # 2 for r, 0.5 for phi
        return mem_bytes / (1024 * 1024)
