"""
Example usage of the flash-attn-triton package.
"""

import torch
from flash_attn_triton import flash_attn_func, flash_attn_qkvpacked_func, FlashAttention

# Set up some example inputs
batch_size = 2
seq_len = 1024
num_heads = 8
head_dim = 64

# Create random query, key, value tensors
q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
k = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
v = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)

# Example 1: Basic unpacked usage
out = flash_attn_func(q, k, v, causal=True)
print(f"Output shape: {out.shape}")

# Example 2: Packed QKV usage
qkv = torch.stack([q, k, v], dim=2)  # Shape: [batch_size, seq_len, 3, num_heads, head_dim]
out_packed = flash_attn_qkvpacked_func(qkv, causal=True)
print(f"Packed output shape: {out_packed.shape}")

# Example 3: Module interface
flash_attn_module = FlashAttention(softmax_scale=1.0 / (head_dim ** 0.5))
out_module = flash_attn_module(q, k, v, causal=True)
print(f"Module output shape: {out_module.shape}")

# Example 4: Multi-query attention (MQA)
# In MQA, the number of heads in keys and values is less than in queries
num_kv_heads = 2
k_mqa = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device='cuda', dtype=torch.float16)
v_mqa = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device='cuda', dtype=torch.float16)

out_mqa = flash_attn_func(q, k_mqa, v_mqa, causal=True)
print(f"MQA output shape: {out_mqa.shape}")
