"""Triton-based implementation of Flash Attention with Flash Attention 2 compatible API."""

__version__ = "0.1.0"

# Import the main interfaces to expose at the package level
from .flash_attn_interface import (
    flash_attn_func,
    flash_attn_qkvpacked_func,
    flash_attn_with_kvcache,
    FlashAttention,
)

# For full compatibility with Flash Attention 2's import structure
# These are aliased to indicate they are not fully supported
from .flash_attn_interface import flash_attn_func as flash_attn_varlen_func
from .flash_attn_interface import flash_attn_qkvpacked_func as flash_attn_varlen_qkvpacked_func
