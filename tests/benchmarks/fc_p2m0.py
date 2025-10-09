import os
import sys

import torch

torch.set_float32_matmul_precision('medium')
torch._dynamo.config.cache_size_limit = 128

if not os.path.exists("clifford-group-equivariant-neural-networks"):
    os.system(
        "git clone https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks.git"
    )

sys.path.append("clifford-group-equivariant-neural-networks")

from algebra.cliffordalgebra import CliffordAlgebra
from models.modules.fcgp import FullyConnectedSteerableGeometricProductLayer

from ops.fc_p2m0 import fused_gelu_fc_sgp_norm_2d
from tests.baselines import gelu_fcgp_norm_2d_torch
from tests.utils import (
    plot_heatmap,
    print_results_table,
    run_sweep,
)


def setup_benchmark(batch_size, num_features):
    """Setup tensors and layers for fc_p2m0 benchmark."""
    algebra = CliffordAlgebra((1, 1))
    sgp = FullyConnectedSteerableGeometricProductLayer(algebra, num_features, num_features).cuda()

    x = torch.randn(4, batch_size, num_features).cuda()
    y = torch.randn(4, batch_size, num_features).cuda()
    weight = sgp.weight.permute(2, 1, 0).contiguous()
    weight_expanded = sgp._get_weight()

    return x, y, weight, weight_expanded, sgp


if __name__ == "__main__":
    assert torch.cuda.is_available()

    path = "tests/benchmarks/results/fc_p2m0"

    results = run_sweep(
        fused_gelu_fc_sgp_norm_2d,
        gelu_fcgp_norm_2d_torch,
        setup_benchmark,
        batch_sizes=[1024, 2048, 4096, 8192],
        num_features_list=[128, 256, 512, 1024],
        n_measure=100
    )

    print_results_table(results, "fc_p2m0")

    plot_heatmap(results, 'speedup_fwd', 'Forward Pass Speedup: Triton vs PyTorch\nFC Cl(1,1)',
                 path + '/speedup/fwd.png')
    plot_heatmap(results, 'speedup_fwd_bwd', 'Forward + Backward Pass Speedup: Triton vs PyTorch\nFC Cl(1,1)',
                 path + '/speedup/fwd_bwd.png')
    plot_heatmap(results, 'mem_ratio_fwd', 'Forward Pass Memory Ratio: Fused / PyTorch\nFC Cl(1,1)',
                 path + '/memory/fwd.png', invert_cmap=True)
    plot_heatmap(results, 'mem_ratio_fwd_bwd', 'Forward + Backward Pass Memory Ratio: Fused / PyTorch\nFC Cl(1,1)',
                 path + '/memory/fwd_bwd.png', invert_cmap=True)
