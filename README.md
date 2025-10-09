<div align="center">

# Flash Clifford

<img src="logo.png" alt="Flash Clifford Logo" width="50%">

</div>

`flash-clifford` provides optimized Triton implementations of weighted geometric product and fully connected geometric product for 2D and 3D Euclidean spaces.
The implementation fuses GELU activation, fully-connected/weighted geometric products, and grade-wise RMSNorm into few kernel operations, achieving significant speedups and memory savings over baseline PyTorch implementations that employs matrix multiplication. The spedup is achieved by manually encoding geometric product rules in forward and backward passes, which otherwise is done via multiplication with a sparse matrix (85-99% sparse depending on the dimensionality).

## Performance

Flash Clifford achieves **~10x speedup** and **22-52% memory reduction** compared to PyTorch baseline (both torch compiled):

<table>
<tr>
<td width="50%">

![Forward Runtime Comparison](tests/benchmarks/results/fc_p3m0/speedup/comparison.png)

</td>
<td width="50%">

![Memory Usage Comparison](tests/benchmarks/results/fc_p3m0/memory/comparison.png)

</td>
</tr>
</table>


## Installation

```bash
git clone https://github.com/maxxxzdn/flash-clifford.git
cd flash-clifford
uv pip install torch triton
```

## Usage

```python
import torch
from modules.layer import Layer

# Input: multivectors of shape (8, batch, features)
x = torch.randn(8, 4096, 512).cuda()

# Linear layer: grade-wise liner + weighted GP
layer = Layer(512, 3).cuda()

output = layer(x)
```

## Benchmarking

Run benchmarks with:

```bash
python tests/benchmarks/fc_p3m0.py
```

## Testing

Verify correctness against PyTorch baseline:

```bash
python tests/fc_p3m0.py
```