import math
import torch

from ops import fused_gelu_sgp_norm_2d, fused_gelu_sgp_norm_3d, fused_gelu_fc_sgp_norm_2d, fused_gelu_fc_sgp_norm_3d, P2M0_NUM_PRODUCT_WEIGHTS, P3M0_NUM_PRODUCT_WEIGHTS, P2M0_NUM_GRADES, P3M0_NUM_GRADES

_FUSED_OPS = {
    (2, False): fused_gelu_sgp_norm_2d,
    (2, True): fused_gelu_fc_sgp_norm_2d,
    (3, False): fused_gelu_sgp_norm_3d,
    (3, True): fused_gelu_fc_sgp_norm_3d,
}

_CONFIG = {
    2: {
        'num_product_weights': P2M0_NUM_PRODUCT_WEIGHTS,
        'num_grades': P2M0_NUM_GRADES,
        'weight_expansion': torch.tensor([0, 1, 1, 2], dtype=torch.long),
    },
    3: {
        'num_product_weights': P3M0_NUM_PRODUCT_WEIGHTS,
        'num_grades': P3M0_NUM_GRADES,
        'weight_expansion': torch.tensor([0, 1, 1, 1, 2, 2, 2, 3], dtype=torch.long),
    }
}


class Layer(torch.nn.Module):

    def __init__(self, n_features, dims, normalize=True, use_fc=False):
        super().__init__()

        if dims not in _CONFIG:
            raise ValueError(f"Unsupported dims: {dims}")

        config = _CONFIG[dims]
        self.normalize = normalize
        self.fused_op = _FUSED_OPS[(dims, use_fc)]

        gp_weight_shape = (n_features, config['num_product_weights'])
        self.gp_weight = torch.nn.Parameter(torch.empty(gp_weight_shape))

        linear_weight_shape = (config['num_grades'], n_features, n_features)
        self.linear_weight = torch.nn.Parameter(torch.empty(linear_weight_shape))
        self.linear_bias = torch.nn.Parameter(torch.zeros(1, 1, n_features))

        self.register_buffer("weight_expansion", config['weight_expansion'])

        torch.nn.init.normal_(self.gp_weight, std=1 / (math.sqrt(dims + 1)))
        torch.nn.init.normal_(self.linear_weight, std=1 / math.sqrt(n_features))

    def forward(self, x):
        y = torch.bmm(x, self.linear_weight[self.weight_expansion]) + self.linear_bias
        return self.fused_op(x, y, self.gp_weight, self.normalize)