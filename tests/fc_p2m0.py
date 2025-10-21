import torch

torch.set_float32_matmul_precision('medium')

from ops.fc_p2m0 import fused_gelu_fcgp_norm_2d
from tests.baselines import gelu_fcgp_norm_2d_torch
from tests.utils import run_correctness_test, run_benchmark


if __name__ == "__main__":
    assert torch.cuda.is_available()

    rep = 1000
    batch_size = 4096
    num_features = 512

    x = torch.randn(4, batch_size, num_features).cuda().contiguous()
    y = torch.randn(4, batch_size, num_features).cuda().contiguous()
    weight = torch.randn(10, num_features, num_features).cuda().contiguous()

    run_correctness_test(fused_gelu_fcgp_norm_2d, gelu_fcgp_norm_2d_torch, {'x': x, 'y': y, 'weight': weight})
    run_benchmark(fused_gelu_fcgp_norm_2d, gelu_fcgp_norm_2d_torch, (x, y, weight), rep, verbose=True)