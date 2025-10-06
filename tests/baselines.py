import torch


def mv_gelu(x):
    """Apply GELU activation gated by scalar component."""
    scalar = x[..., [0]]
    gate = 0.5 * (1 + torch.erf(scalar * 0.7071067811865475))
    return x * gate


def mv_rmsnorm_2d(x, eps=1e-6):
    """RMS normalization for 2D Clifford algebra (scalar, vector, pseudoscalar)."""
    scalar = x[..., [0]]
    vector = x[..., [1, 2]]
    pseudoscalar = x[..., [3]]

    scalar_rms = (scalar.pow(2).mean(dim=1, keepdim=True) + eps).sqrt()
    scalar = scalar / scalar_rms

    vector_norm = vector.norm(dim=2, keepdim=True)
    vector_rms = (vector_norm.pow(2).mean(dim=1, keepdim=True) + eps).sqrt()
    vector = vector / vector_rms

    pseudoscalar_rms = (pseudoscalar.pow(2).mean(dim=1, keepdim=True) + eps).sqrt()
    pseudoscalar = pseudoscalar / pseudoscalar_rms

    return torch.cat([scalar, vector, pseudoscalar], dim=-1)


def mv_rmsnorm_3d(x, eps=1e-6):
    """RMS normalization for 3D Clifford algebra (scalar, vector, bivector, pseudoscalar)."""
    scalar = x[..., [0]]
    vector = x[..., [1, 2, 3]]
    bivector = x[..., [4, 5, 6]]
    pseudoscalar = x[..., [7]]

    scalar_rms = (scalar.pow(2).mean(dim=1, keepdim=True) + eps).sqrt()
    scalar = scalar / scalar_rms

    vector_norm = vector.norm(dim=2, keepdim=True)
    vector_rms = (vector_norm.pow(2).mean(dim=1, keepdim=True) + eps).sqrt()
    vector = vector / vector_rms

    bivector_norm = bivector.norm(dim=2, keepdim=True)
    bivector_rms = (bivector_norm.pow(2).mean(dim=1, keepdim=True) + eps).sqrt()
    bivector = bivector / bivector_rms

    pseudoscalar_rms = (pseudoscalar.pow(2).mean(dim=1, keepdim=True) + eps).sqrt()
    pseudoscalar = pseudoscalar / pseudoscalar_rms

    return torch.cat([scalar, vector, bivector, pseudoscalar], dim=-1)


@torch.compile
def gelu_sgp_norm_2d_torch(x, y, weight_expanded):
    """Geometric product layer with GELU activation and RMS normalization in 2D Clifford algebra."""
    x = mv_gelu(x)
    y = mv_gelu(y)
    o = torch.einsum("bni, nijk, bnk -> bnj", x, weight_expanded, y)
    o = mv_rmsnorm_2d(o)
    return o


@torch.compile
def gelu_sgp_norm_3d_torch(x, y, weight_expanded):
    """Geometric product layer with GELU activation and RMS normalization in 3D Clifford algebra."""
    x = mv_gelu(x)
    y = mv_gelu(y)
    o = torch.einsum("bni, nijk, bnk -> bnj", x, weight_expanded, y)
    o = mv_rmsnorm_3d(o)
    return o


@torch.compile
def gelu_fcgp_norm_2d_torch(x, y, weight_expanded):
    """Fully connected geometric product layer with GELU activation and RMS normalization in 2D Clifford algebra."""
    x = mv_gelu(x)
    y = mv_gelu(y)
    o = torch.einsum("bni, mnijk, bnk -> bmj", x, weight_expanded, y)
    o = mv_rmsnorm_2d(o)
    return o


@torch.compile
def gelu_fcgp_norm_3d_torch(x, y, weight_expanded):
    """Fully connected geometric product layer with GELU activation and RMS normalization in 3D Clifford algebra."""
    x = mv_gelu(x)
    y = mv_gelu(y)
    o = torch.einsum("bni, mnijk, bnk -> bmj", x, weight_expanded, y)
    o = mv_rmsnorm_3d(o)
    return o


