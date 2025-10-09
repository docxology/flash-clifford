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
from models.modules.fcgp import FullyConnectedSteerableGeometricProductLayer

from ops.fc_p3m0 import fused_gelu_fc_sgp_norm_3d
from tests.utils import run_correctness_tests, run_benchmarks
from tests.baselines import gelu_fcgp_norm_3d_torch


if __name__ == "__main__":
    assert torch.cuda.is_available()

    n_measure = 200
    batch_size = 4096
    num_features = 512

    algebra = CliffordAlgebra((1, 1, 1))
    sgp = FullyConnectedSteerableGeometricProductLayer(algebra, num_features, num_features).cuda()

    x = torch.randn(8, batch_size, num_features).cuda()
    y = torch.randn(8, batch_size, num_features).cuda()
    sgp_weight = sgp.weight.permute(2, 1, 0).contiguous()

    sgp_ret, weight_triton = run_correctness_tests(
        fused_gelu_fc_sgp_norm_3d,
        gelu_fcgp_norm_3d_torch,
        x, y,
        sgp_weight,
        sgp
    )

    print(
        f"grad_weight max diff: {(sgp_ret.weight.grad - weight_triton.grad.permute(2, 1, 0)).abs().max().item():.1e}"
        + (" ✔" if torch.allclose(sgp_ret.weight.grad, weight_triton.grad.permute(2, 1, 0), atol=1e-1) else " ✘")
    )

    run_benchmarks(fused_gelu_fc_sgp_norm_3d, gelu_fcgp_norm_3d_torch, x, y, weight_triton, sgp, 100, 5)