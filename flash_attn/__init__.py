"""Compatibility layer for Flash Attention 2"""

# Import from our implementation
from flash_attn_triton.flash_attn_interface import (
    flash_attn_func,
    flash_attn_qkvpacked_func,
    flash_attn_kvpacked_func,
    FlashAttention,
)

