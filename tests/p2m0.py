import os
import sys

import torch

torch.set_float32_matmul_precision('medium')

if not os.path.exists("clifford-group-equivariant-neural-networks"):
    os.system(
        "git clone https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks.git"
    )

sys.path.append("clifford-group-equivariant-neural-networks")

from algebra.cliffordalgebra import CliffordAlgebra
from models.modules.gp import SteerableGeometricProductLayer

from ops.p2m0 import fused_gelu_sgp_norm_2d
from tests.utils import run_correctness_tests, run_benchmarks
from tests.baselines import gelu_sgp_norm_2d_torch


if __name__ == "__main__":
    assert torch.cuda.is_available()

    n_measure = 1000
    batch_size = 4096
    num_features = 512

    algebra = CliffordAlgebra((1, 1))
    sgp = SteerableGeometricProductLayer(algebra, num_features).cuda()

    x = torch.randn(4, batch_size, num_features).cuda()
    y = torch.randn(4, batch_size, num_features).cuda()
    sgp_weight = sgp.weight

    sgp_ret, weight_triton = run_correctness_tests(
        fused_gelu_sgp_norm_2d,
        gelu_sgp_norm_2d_torch,
        x, y,
        sgp_weight,
        sgp
    )

    print(
        f"grad_weight max diff: {(sgp_ret.weight.grad - weight_triton.grad).abs().max().item():.1e}"
        + (" ✔" if torch.allclose(sgp_ret.weight.grad, weight_triton.grad, atol=1e-2) else " ✘")
    )
    
    run_benchmarks(fused_gelu_sgp_norm_2d, gelu_sgp_norm_2d_torch, x, y, weight_triton, sgp, 100, 5)