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

from ops.p2m0 import WeightedGeluGeometricProductNorm2D
from tests.utils import mv_gelu, mv_rmsnorm_2d, run_correctness_tests, run_benchmarks


@torch.compile
def gelu_sgp_norm_torch(x, y, sgp):
    x = mv_gelu(x)
    y = mv_gelu(y)
    weight_expanded = sgp._get_weight()
    o = torch.einsum("bni, nijk, bnk -> bnj", x, weight_expanded, y)
    o = mv_rmsnorm_2d(o)
    return o


@torch.compile
def gelu_sgp_norm_triton(x, y, weight):
    return WeightedGeluGeometricProductNorm2D.apply(
        x, y, weight, True
    )


if __name__ == "__main__":
    assert torch.cuda.is_available()

    n_measure = 1000
    batch_size = 4096
    num_features = 1024

    algebra = CliffordAlgebra((1, 1))
    sgp = SteerableGeometricProductLayer(algebra, num_features).cuda()

    x = torch.randn(batch_size, num_features, 4).cuda()
    y = torch.randn(batch_size, num_features, 4).cuda()
    sgp_weight = sgp.weight

    sgp_ret, weight_triton = run_correctness_tests(
        gelu_sgp_norm_triton,
        gelu_sgp_norm_torch,
        x, y,
        sgp_weight,
        sgp
    )

    print(
        f"grad_weight max diff: {(sgp_ret.weight.grad - weight_triton.grad).abs().max().item():.1e}"
        + (" ✔" if torch.allclose(sgp_ret.weight.grad, weight_triton.grad, atol=1e-2) else " ✘")
    )

    run_benchmarks(
        gelu_sgp_norm_triton,
        gelu_sgp_norm_torch,
        x, y,
        sgp_weight,
        sgp,
        n_measure
    )
