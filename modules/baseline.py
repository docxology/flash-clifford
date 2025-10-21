import math
import torch

from ops import P2M0_NUM_PRODUCT_WEIGHTS, P3M0_NUM_PRODUCT_WEIGHTS, P2M0_NUM_GRADES, P3M0_NUM_GRADES
from tests.baselines import gelu_sgp_norm_2d_torch, gelu_sgp_norm_3d_torch, gelu_fcgp_norm_2d_torch, gelu_fcgp_norm_3d_torch

_FUSED_OPS = {
    (2, False): gelu_sgp_norm_2d_torch,
    (2, True): gelu_fcgp_norm_2d_torch,
    (3, False): gelu_sgp_norm_3d_torch,
    (3, True): gelu_fcgp_norm_3d_torch,
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
    """ 
    Linear layer: grade-wise linear + weighted GP + GELU + LayerNorm.
    Metric-specific implementation of https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks/blob/8482b06b71712dcea2841ebe567d37e7f8432d27/models/nbody_cggnn.py#L47
    
    Args:
        n_features: number of features.
        dims: 2 or 3, dimension of the space.
        normalize: whether to apply layer normalization at the end.
        use_fc: whether to use fully connected GP weights.
            True: weight has shape (n_features, n_features, num_product_weights).
            False: weight has shape (n_features, num_product_weights).
    """
    def __init__(self, n_features, dims, normalize=True, use_fc=False):
        super().__init__()

        if dims not in _CONFIG:
            raise ValueError(f"Unsupported dims: {dims}")

        config = _CONFIG[dims]
        self.normalize = normalize
        self.fused_op = _FUSED_OPS[(dims, use_fc)]

        if use_fc:
            gp_weight_shape = (config['num_product_weights'], n_features, n_features)
        else:
            gp_weight_shape = (n_features, config['num_product_weights'])
            
        self.gp_weight = torch.nn.Parameter(torch.empty(gp_weight_shape))

        linear_weight_shape = (config['num_grades'], n_features, n_features)
        self.linear_weight = torch.nn.Parameter(torch.empty(linear_weight_shape))
        self.linear_bias = torch.nn.Parameter(torch.zeros(1, 1, n_features))

        self.register_buffer("weight_expansion", config['weight_expansion'])

        torch.nn.init.normal_(self.gp_weight, std=1 / (math.sqrt(dims + 1)))
        torch.nn.init.normal_(self.linear_weight, std=1 / math.sqrt(n_features) if use_fc else 1 / math.sqrt(n_features * (dims + 1)))

    def forward(self, x):
        y = torch.bmm(x, self.linear_weight[self.weight_expansion]) + self.linear_bias
        return self.fused_op(x, y, self.gp_weight, self.normalize)