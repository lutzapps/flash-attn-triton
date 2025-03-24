"""Compatibility layer for Flash Attention 2"""

# Import from our implementation
from flash_attn_triton.flash_attn_interface import (
    flash_attn_func,
    flash_attn_qkvpacked_func,
    flash_attn_with_kvcache,
    FlashAttention,
    # Add these aliases to match Flash Attention 2
    flash_attn_func as flash_attn_varlen_func,
    flash_attn_qkvpacked_func as flash_attn_varlen_qkvpacked_func,
)
