"""
Main interfaces for the Triton-based Flash Attention implementation.
"""

import torch
import math
from typing import Optional, Tuple, Union

# Import the attention function from the Triton kernel implementation
from .triton_kernel.attention_kernel import attention

def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                    window_size=(-1, -1), alibi_slopes=None, deterministic=False,
                    softcap=0.0,
                    return_attn_probs=False):
    """Flash Attention implementation using Triton kernel.
    
    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        dropout_p: float. Dropout probability (not supported in this implementation).
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask.
        window_size: (left, right). If not (-1, -1), implements sliding window local attention
            (not supported in this implementation).
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. For ALiBi attention bias
            (not supported in this implementation).
        deterministic: bool. Whether to use the deterministic implementation of the backward pass
            (not supported in this implementation).
        return_attn_probs: bool. Whether to return attention probabilities (not supported in this
            implementation).
            
    Return:
        out: (batch_size, seqlen, nheads, headdim).
    """
    # Check unsupported features
    if window_size != (-1, -1):
        raise ValueError("Warning: sliding window attention is not supported in this Triton implementation")
    if alibi_slopes is not None:
        raise ValueError("Warning: ALiBi is not supported in this Triton implementation")
    if softcap != 0.0:
        raise ValueError("Warning: Softcap is not supported in this Triton implementation")
    if deterministic:
        print("Warning: deterministic backward pass is not built into this Triton implementation")
    if return_attn_probs:
        print("Warning: returning attention probabilities is not built into this Triton implementation")
    
        
    # Ensure tensors are contiguous
    q = q.contiguous() if not q.is_contiguous() else q
    k = k.contiguous() if not k.is_contiguous() else k
    v = v.contiguous() if not v.is_contiguous() else v
    
    # Validate input shapes
    batch_size, seq_len_q, n_heads_q, head_dim = q.shape
    _, seq_len_k, n_heads_k, _ = k.shape
    
    # Check if we're dealing with MQA/GQA (not fully supported)
    if n_heads_q != n_heads_k:
        if n_heads_q % n_heads_k != 0:
            raise ValueError(
                f"Number of heads in Q ({n_heads_q}) must be divisible by number of heads in K/V ({n_heads_k})"
            )
        print("Warning: MQA/GQA will use simple reshaping which may not match Flash Attention's implementation")
        # Simple implementation: repeat k and v
        k = k.repeat_interleave(n_heads_q // n_heads_k, dim=2)
        v = v.repeat_interleave(n_heads_q // n_heads_k, dim=2)
    
    # Transpose q, k, v from [batch, seq_len, heads, head_dim] to [batch, heads, seq_len, head_dim]
    # which is what the Triton implementation expects
    q = q.transpose(1, 2).contiguous()  # Ensure contiguity after transpose
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    
    # Compute softmax scale if not provided
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    
    # Call the Triton implementation
    out = attention(q, k, v, causal, softmax_scale)
    
    # Transpose the output back to the expected shape [batch, seq_len, heads, head_dim]
    out = out.transpose(1, 2).contiguous()  # Ensure output is contiguous

    if return_attn_probs:
        return out, None, None
    else:
        return out

def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False,
                              window_size=(-1, -1), alibi_slopes=None, deterministic=False, softcap=0.0,
                              return_attn_probs=False):
    """Flash Attention for packed QKV using Triton kernel.
    
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, headdim)
        dropout_p: float. Dropout probability (not supported in this implementation).
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask.
        window_size: (left, right). If not (-1, -1), implements sliding window local attention
            (not supported in this implementation).
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. For ALiBi attention bias
            (not supported in this implementation).
        deterministic: bool. Whether to use the deterministic implementation of the backward pass
            (not supported in this implementation).
        return_attn_probs: bool. Whether to return attention probabilities (not supported in this
            implementation).
            
    Return:
        out: (batch_size, seqlen, nheads, headdim).
    """
    # Ensure input is contiguous
    qkv = qkv.contiguous() if not qkv.is_contiguous() else qkv
    
    # Unpack qkv
    batch_size, seqlen, _, n_heads, head_dim = qkv.shape
    q, k, v = qkv.unbind(dim=2)
    
    # Call the unpacked version
    return flash_attn_func(
        q, k, v, dropout_p, softmax_scale, causal, window_size, alibi_slopes, deterministic, softcap, return_attn_probs
    )


def flash_attn_kvpacked_func(q, kv, dropout_p=0.0, softmax_scale=None, causal=False,
                             window_size=(-1, -1), alibi_slopes=None, deterministic=False,
                             softcap=0.0, return_attn_probs=False):
    """Flash Attention for packed KV using Triton kernel.
    
    Arguments:
        q: (batch_size, seqlen_q, nheads, headdim)
        kv: (batch_size, seqlen_k, 2, nheads_k, headdim)
        dropout_p: float. Dropout probability (not supported in this implementation).
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask.
        window_size: (left, right). If not (-1, -1), implements sliding window local attention
            (not supported in this implementation).
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. For ALiBi attention bias
            (not supported in this implementation).
        deterministic: bool. Whether to use the deterministic implementation of the backward pass
            (not supported in this implementation).
            
    Return:
        out: (batch_size, seqlen_q, nheads, headdim).
    """
    # Ensure KV is contiguous
    kv = kv.contiguous() if not kv.is_contiguous() else kv
    
    # Unpack kv
    k, v = kv.unbind(dim=2)
    
    # Call the unpacked version
    return flash_attn_func(
        q, k, v, dropout_p, softmax_scale, causal, window_size, alibi_slopes, deterministic, softcap, return_attn_probs
    )


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens=None,
    cache_batch_idx=None,
    block_table=None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    rotary_interleaved=True,
    alibi_slopes=None,
):
    """Flash attention with KV cache, limited implementation.
    
    Note: This implementation has significant limitations compared to the full Flash Attention 2 API.
    It doesn't support many features like in-place KV cache updates, paged KV cache, or rotary embeddings.
    
    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim)
        v_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim)
        k: Optional new keys to be appended to k_cache
        v: Optional new values to be appended to v_cache
        rotary_cos, rotary_sin: Rotary embeddings (not supported)
        cache_seqlens: Sequence lengths for the cache (not supported)
        cache_batch_idx: Batch indices for the cache (not supported)
        block_table: For paged KV cache (not supported)
        softmax_scale: Softmax scale
        causal: Whether to apply causal masking
        window_size: For local attention (not supported)
        rotary_interleaved: Rotary embedding style (not supported)
        alibi_slopes: For ALiBi (not supported)
        
    Returns:
        out: (batch_size, seqlen, nheads, headdim)
    """
    print("Warning: flash_attn_with_kvcache is a limited implementation that doesn't support most KV cache features")
    
    # Check if we're using features not supported by this implementation
    if rotary_cos is not None or rotary_sin is not None:
        print("Warning: rotary embeddings are not supported in this implementation")
    if cache_seqlens is not None:
        print("Warning: cache_seqlens is not supported in this implementation")
    if cache_batch_idx is not None:
        print("Warning: cache_batch_idx is not supported in this implementation")
    if block_table is not None:
        print("Warning: paged KV cache is not supported in this implementation")
    if window_size != (-1, -1):
        print("Warning: sliding window attention is not supported in this implementation")
    if alibi_slopes is not None:
        print("Warning: ALiBi is not supported in this implementation")
    
    # Ensure inputs are contiguous
    q = q.contiguous() if not q.is_contiguous() else q
    k_cache = k_cache.contiguous() if not k_cache.is_contiguous() else k_cache
    v_cache = v_cache.contiguous() if not v_cache.is_contiguous() else v_cache
    
    # Basic implementation without proper KV cache updates
    batch_size = q.shape[0]
    
    # If we have new k and v, concatenate them to the cache (without in-place updates)
    if k is not None and v is not None:
        # Ensure new k, v are contiguous
        k = k.contiguous() if not k.is_contiguous() else k
        v = v.contiguous() if not v.is_contiguous() else v
        
        if k_cache.shape[0] != batch_size or v_cache.shape[0] != batch_size:
            raise ValueError("Batch size mismatch between query and KV cache")
        
        # Simple concatenation (not in-place)
        k_full = torch.cat([k_cache, k], dim=1)
        v_full = torch.cat([v_cache, v], dim=1)
    else:
        k_full = k_cache
        v_full = v_cache
    
    # Run attention with the full KV tensors
    return flash_attn_func(q, k_full, v_full, dropout_p=0.0, softmax_scale=softmax_scale, causal=causal)

class FlashAttention(torch.nn.Module):
    """
    Module implementation of Flash Attention using Triton kernel.
    
    Note: This is a limited implementation that doesn't support all features of Flash Attention 2.
    """
    def __init__(self, attention_dropout=0.0, softmax_scale=None):
        super().__init__()
        self.dropout_p = attention_dropout
        self.softmax_scale = softmax_scale
        if attention_dropout > 0.0:
            print("Warning: dropout is not supported in this Triton implementation")
    
    def forward(self, q, k, v, causal=False, attn_bias=None):
        """
        Forward pass.
        
        Args:
            q: Query tensor of shape (batch_size, seqlen_q, num_heads, head_dim)
            k: Key tensor of shape (batch_size, seqlen_k, num_heads, head_dim)
            v: Value tensor of shape (batch_size, seqlen_k, num_heads, head_dim)
            causal: If True, applies causal masking
            attn_bias: Optional attention bias (not supported in this implementation)
            
        Returns:
            output: Attention output of shape (batch_size, seqlen_q, num_heads, head_dim)
        """
        if attn_bias is not None:
            print("Warning: attention bias is not supported in this Triton implementation")
        
        return flash_attn_func(
            q, k, v, 
            dropout_p=self.dropout_p,
            softmax_scale=self.softmax_scale,
            causal=causal
        )
