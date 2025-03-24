"""
Compare outputs from our Flash Attention Triton implementation with the original Flash Attention 2.

Usage:
    python -m tests.compare_implementations
"""

import torch
import numpy as np
import importlib.util
import os
import sys

def has_package(package_name):
    """Check if a package is installed."""
    return importlib.util.find_spec(package_name) is not None

# Check if both implementations are available
if not has_package('flash_attn_triton'):
    print("ERROR: flash_attn_triton is not installed. Please install it first.")
    sys.exit(1)

if not has_package('flash_attn'):
    print("WARNING: Original flash_attn is not installed. Will only run our implementation.")
    original_available = False
else:
    original_available = True

# Import our implementation
from flash_attn_triton import flash_attn_func as our_flash_attn_func

# Import original implementation if available
if original_available:
    # Temporarily remove our flash_attn from sys.modules to avoid import conflicts
    if 'flash_attn' in sys.modules:
        saved_flash_attn = sys.modules['flash_attn']
        del sys.modules['flash_attn']
    
    # Import the original implementation
    from flash_attn import flash_attn_func as original_flash_attn_func
    
    # Restore our flash_attn if it was saved
    if 'saved_flash_attn' in locals():
        sys.modules['flash_attn'] = saved_flash_attn

def create_test_inputs(batch_size=2, seq_len=1024, num_heads=8, head_dim=64, dtype=torch.float16, device='cuda'):
    """Create test inputs for comparison."""
    torch.manual_seed(42)  # For reproducibility
    
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    
    return q, k, v

def compare_outputs(causal=True, softmax_scale=None, atol=1e-3, rtol=1e-3):
    """Compare outputs from both implementations."""
    print(f"Testing {'causal' if causal else 'non-causal'} attention:")
    
    q, k, v = create_test_inputs()
    
    # Run our implementation
    try:
        our_output = our_flash_attn_func(q, k, v, causal=causal, softmax_scale=softmax_scale)
        print(f"Our implementation output shape: {our_output.shape}")
    except Exception as e:
        print(f"ERROR in our implementation: {str(e)}")
        our_output = None
    
    # Run original implementation if available
    if original_available:
        try:
            original_output = original_flash_attn_func(q, k, v, causal=causal, softmax_scale=softmax_scale)
            print(f"Original implementation output shape: {original_output.shape}")
        except Exception as e:
            print(f"ERROR in original implementation: {str(e)}")
            original_output = None
        
        # Compare outputs
        if our_output is not None and original_output is not None:
            max_diff = torch.max(torch.abs(our_output - original_output)).item()
            if torch.allclose(our_output, original_output, atol=atol, rtol=rtol):
                print(f"✅ Outputs match! Maximum absolute difference: {max_diff:.6f}")
            else:
                print(f"❌ Outputs differ! Maximum absolute difference: {max_diff:.6f}")
            
            # Compare gradients
            sum_output = our_output.sum()
            sum_output.backward(retain_graph=True)
            our_grads = [q.grad.clone(), k.grad.clone(), v.grad.clone()]
            q.grad, k.grad, v.grad = None, None, None
            
            sum_output_original = original_output.sum()
            sum_output_original.backward()
            original_grads = [q.grad, k.grad, v.grad]
            
            for i, (our_grad, original_grad) in enumerate(zip(our_grads, original_grads)):
                param_name = ["q", "k", "v"][i]
                grad_max_diff = torch.max(torch.abs(our_grad - original_grad)).item()
                if torch.allclose(our_grad, original_grad, atol=atol, rtol=rtol):
                    print(f"✅ {param_name} gradients match! Maximum absolute difference: {grad_max_diff:.6f}")
                else:
                    print(f"❌ {param_name} gradients differ! Maximum absolute difference: {grad_max_diff:.6f}")
    print()

def run_tests():
    """Run all test cases."""
    print("=" * 50)
    print("Comparing Flash Attention Implementations")
    print("=" * 50)
    
    # Test different configurations
    compare_outputs(causal=True)
    compare_outputs(causal=False)
    compare_outputs(causal=True, softmax_scale=0.125)

if __name__ == "__main__":
    run_tests()
