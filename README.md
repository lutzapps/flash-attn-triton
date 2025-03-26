# Flash Attention Triton

This repository provides a Triton-based implementation of the Flash Attention algorithm with a Flash Attention 2 compatible API. It allows for a drop-in replacement of the original Flash Attention 2 package for supported functionality.

## Installation

You can install the package directly from GitHub:

```bash
pip install git+https://github.com/rationalism/flash-attn-triton.git
```

Or from PyPI:

```bash
pip install flash-attn-triton
```

## Requirements

- PyTorch 2.6 or later
- Triton 3.2 or later
- CUDA-compatible GPU (compute capability 7.0+)

## Usage

The API is designed to be compatible with Flash Attention 2. You can use it in the same way:

```python
from flash_attn_triton import flash_attn_func, flash_attn_qkvpacked_func, FlashAttention

# Basic usage
out = flash_attn_func(q, k, v, causal=True)

# Packed QKV
out = flash_attn_qkvpacked_func(qkv, causal=True)

# Module interface
flash_attn = FlashAttention()
out = flash_attn(q, k, v, causal=True)
```

## Currently Supported Features

- Basic attention mechanism (forward and backward)
- Causal masking
- Softmax scaling
- Basic MQA/GQA support (via tensor repetition)
- Head dims 16, 32, 64, 128

## Limitations

This implementation does not currently support:

- Non-causal attention for sequence lengths not divisible by 128
- Dropout (in progress)
- Attention bias
- Sliding window attention
- ALiBi
- KV caching with in-place updates
- Softcapping
- Deterministic backward pass

## Benchmarks

# RTX 3090 (Ampere)

```
fused-attention-batch4-head32-d64-fwd-causal=True-dropout=0.0:
     N_CTX  Triton [FP16]
0   1024.0      48.049147
1   2048.0      61.062769
2   4096.0      68.363188
3   8192.0      70.768167
4  16384.0      72.332634
fused-attention-batch4-head32-d64-fwd-causal=False-dropout=0.0:
     N_CTX  Triton [FP16]
0   1024.0      60.190653
1   2048.0      71.126662
2   4096.0      69.049310
3   8192.0      74.579215
4  16384.0      73.911621
fused-attention-batch4-head32-d64-bwd-causal=True-dropout=0.0:
     N_CTX  Triton [FP16]
0   1024.0      33.531732
1   2048.0      40.884683
2   4096.0      45.627974
3   8192.0      47.449394
4  16384.0      48.993511
fused-attention-batch4-head32-d64-bwd-causal=False-dropout=0.0:
     N_CTX  Triton [FP16]
0   1024.0      42.834959
1   2048.0      46.382862
2   4096.0      49.984253
3   8192.0      51.358497
4  16384.0      49.913040
```

## Acknowledgements

This implementation is based on the Triton attention implementation from the original Flash Attention 2 repository by TriDao and the Triton tutorial on fused attention.

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.
