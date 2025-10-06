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

from ops.fc_p3m0 import FullyConnectedGeluGeometricProductNorm3D
from tests.utils import mv_gelu, mv_rmsnorm_3d, run_correctness_tests, run_benchmarks


@torch.compile
def gelu_sgp_norm_torch(x, y, sgp):
    x = mv_gelu(x)
    y = mv_gelu(y)
    weight_expanded = sgp._get_weight()
    o = torch.einsum("bni, mnijk, bnk -> bmj", x, weight_expanded, y)
    o = mv_rmsnorm_3d(o)
    return o


@torch.compile
def gelu_sgp_norm_triton(x, y, weight):
    return FullyConnectedGeluGeometricProductNorm3D.apply(
        x, y, weight, True
    )


if __name__ == "__main__":
    assert torch.cuda.is_available()

    with torch.no_grad():
        n_measure = 200
        batch_size = 4096
        num_features = 512

        algebra = CliffordAlgebra((1, 1, 1))
        sgp = FullyConnectedSteerableGeometricProductLayer(algebra, num_features, num_features).cuda()

        x = torch.randn(batch_size, num_features, 8).cuda()
        y = torch.randn(batch_size, num_features, 8).cuda()
        sgp_weight = sgp.weight.permute(2, 1, 0).contiguous()

    sgp_ret, weight_triton = run_correctness_tests(
        gelu_sgp_norm_triton,
        gelu_sgp_norm_torch,
        x, y,
        sgp_weight,
        sgp
    )

    print(
        f"grad_weight max diff: {(sgp_ret.weight.grad - weight_triton.grad.permute(2, 1, 0)).abs().max().item():.1e}"
        + (" ✔" if torch.allclose(sgp_ret.weight.grad, weight_triton.grad.permute(2, 1, 0), atol=1e-2) else " ✘")
    )

    run_benchmarks(
        gelu_sgp_norm_triton,
        gelu_sgp_norm_torch,
        x, y,
        sgp_weight,
        sgp,
        n_measure
    )
