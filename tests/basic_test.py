import torch
from flash_attn_triton import attention

def test_attention():
    # Create sample inputs
    batch_size, num_heads, seq_len, head_dim = 2, 4, 1024, 64
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    
    # Test the attention function
    out = attention(q, k, v, causal=True)
    
    # Check output shape
    assert out.shape == (batch_size, num_heads, seq_len, head_dim)
    
    # Check if gradients flow
    out.sum().backward()
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None

if __name__ == "__main__":
    test_attention()
