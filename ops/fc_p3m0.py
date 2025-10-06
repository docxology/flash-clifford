import torch
import triton
import triton.language as tl

MV_DIM = 8
NUM_GRADES = 4
NUM_PRODUCT_WEIGHTS = 20
WEIGHT_EXPANSION = [0, 1, 1, 1, 2, 2, 2, 3, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 16, 17, 17, 17, 18, 18, 18, 19]
EPS = 1e-6


@triton.jit
def compute_gelu_gate(x):
    """Compute the GELU gate Φ(x) := 0.5 * (1 + erf(x / sqrt(2)))"""
    return 0.5 * (1 + tl.erf(x.to(tl.float32) * 0.7071067811865475)).to(x.dtype)


@triton.jit
def compute_gelu_gate_grad(x):
    """Compute the gradient of the GELU gate = 1/sqrt(2pi) * exp(-x^2/2)"""
    return 0.3989422804 * tl.exp(-0.5 * x * x)


@triton.jit
def gelu_pairwise_kernel_fwd(
    x_ptr,
    y_ptr,
    pairwise_ptr,
    batch_size: tl.constexpr,
    n_features: tl.constexpr,
    BATCH_BLOCK: tl.constexpr,
    FEATURE_BLOCK: tl.constexpr,
    MV_DIM: tl.constexpr,
):
    """
    Apply GELU non-linearity to inputs and compute pairwise products for geometric product.
    """
    batch_block_id = tl.program_id(axis=0)
    thread_block_id = tl.program_id(axis=1)

    batch_ids = batch_block_id*BATCH_BLOCK + tl.arange(0, BATCH_BLOCK)
    feature_ids = thread_block_id*FEATURE_BLOCK + tl.arange(0, FEATURE_BLOCK)

    batch_mask = batch_ids < batch_size
    feature_mask = feature_ids < n_features
    batch_feature_mask = batch_mask[:, None] & feature_mask[None, :]

    feature_offset = batch_ids[:, None] * n_features * MV_DIM + feature_ids[None, :] * MV_DIM
    pairwise_offset = batch_ids[:, None] * n_features + feature_ids[None, :]

    x0 = tl.load(x_ptr + feature_offset + 0, mask=batch_feature_mask)
    x1 = tl.load(x_ptr + feature_offset + 1, mask=batch_feature_mask)
    x2 = tl.load(x_ptr + feature_offset + 2, mask=batch_feature_mask)
    x3 = tl.load(x_ptr + feature_offset + 3, mask=batch_feature_mask)
    x4 = tl.load(x_ptr + feature_offset + 4, mask=batch_feature_mask)
    x5 = tl.load(x_ptr + feature_offset + 5, mask=batch_feature_mask)
    x6 = tl.load(x_ptr + feature_offset + 6, mask=batch_feature_mask)
    x7 = tl.load(x_ptr + feature_offset + 7, mask=batch_feature_mask)

    y0 = tl.load(y_ptr + feature_offset + 0, mask=batch_feature_mask)
    y1 = tl.load(y_ptr + feature_offset + 1, mask=batch_feature_mask)
    y2 = tl.load(y_ptr + feature_offset + 2, mask=batch_feature_mask)
    y3 = tl.load(y_ptr + feature_offset + 3, mask=batch_feature_mask)
    y4 = tl.load(y_ptr + feature_offset + 4, mask=batch_feature_mask)
    y5 = tl.load(y_ptr + feature_offset + 5, mask=batch_feature_mask)
    y6 = tl.load(y_ptr + feature_offset + 6, mask=batch_feature_mask)
    y7 = tl.load(y_ptr + feature_offset + 7, mask=batch_feature_mask)

    gate_x = compute_gelu_gate(x0)
    gate_y = compute_gelu_gate(y0)

    x0 = x0 * gate_x
    x1 = x1 * gate_x
    x2 = x2 * gate_x
    x3 = x3 * gate_x
    x4 = x4 * gate_x
    x5 = x5 * gate_x
    x6 = x6 * gate_x
    x7 = x7 * gate_x

    y0 = y0 * gate_y
    y1 = y1 * gate_y
    y2 = y2 * gate_y
    y3 = y3 * gate_y
    y4 = y4 * gate_y
    y5 = y5 * gate_y
    y6 = y6 * gate_y
    y7 = y7 * gate_y

    p0 = x0*y0
    p1 = x0*y1
    p2 = x0*y2
    p3 = x0*y3
    p4 = x0*y4
    p5 = x0*y5
    p6 = x0*y6
    p7 = x0*y7
    p8 = x1*y1 + x2*y2 + x3*y3
    p9 = x1*y0
    p10 = x2*y0
    p11 = x3*y0
    p12 = -x2*y4 - x3*y5
    p13 = x1*y4 - x3*y6
    p14 = x1*y5 + x2*y6
    p15 = x1*y2 - x2*y1
    p16 = x1*y3 - x3*y1
    p17 = x2*y3 - x3*y2
    p18 = x3*y7
    p19 = -x2*y7
    p20 = x1*y7
    p21 = x1*y6 - x2*y5 + x3*y4
    p22 = -x4*y4 - x5*y5 - x6*y6
    p23 = x4*y2 + x5*y3
    p24 = -x4*y1 + x6*y3
    p25 = -x5*y1 - x6*y2
    p26 = -x6*y7
    p27 = x5*y7
    p28 = -x4*y7
    p29 = x4*y0
    p30 = x5*y0
    p31 = x6*y0
    p32 = -x5*y6 + x6*y5
    p33 = x4*y6 - x6*y4
    p34 = -x4*y5 + x5*y4
    p35 = x4*y3 - x5*y2 + x6*y1
    p36 = -x7*y7
    p37 = -x7*y6
    p38 = x7*y5
    p39 = -x7*y4
    p40 = x7*y3
    p41 = -x7*y2
    p42 = x7*y1
    p43 = x7*y0

    tl.store(pairwise_ptr + 0*batch_size*n_features + pairwise_offset, p0, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 1*batch_size*n_features + pairwise_offset, p1, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 2*batch_size*n_features + pairwise_offset, p2, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 3*batch_size*n_features + pairwise_offset, p3, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 4*batch_size*n_features + pairwise_offset, p4, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 5*batch_size*n_features + pairwise_offset, p5, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 6*batch_size*n_features + pairwise_offset, p6, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 7*batch_size*n_features + pairwise_offset, p7, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 8*batch_size*n_features + pairwise_offset, p8, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 9*batch_size*n_features + pairwise_offset, p9, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 10*batch_size*n_features + pairwise_offset, p10, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 11*batch_size*n_features + pairwise_offset, p11, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 12*batch_size*n_features + pairwise_offset, p12, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 13*batch_size*n_features + pairwise_offset, p13, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 14*batch_size*n_features + pairwise_offset, p14, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 15*batch_size*n_features + pairwise_offset, p15, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 16*batch_size*n_features + pairwise_offset, p16, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 17*batch_size*n_features + pairwise_offset, p17, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 18*batch_size*n_features + pairwise_offset, p18, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 19*batch_size*n_features + pairwise_offset, p19, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 20*batch_size*n_features + pairwise_offset, p20, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 21*batch_size*n_features + pairwise_offset, p21, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 22*batch_size*n_features + pairwise_offset, p22, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 23*batch_size*n_features + pairwise_offset, p23, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 24*batch_size*n_features + pairwise_offset, p24, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 25*batch_size*n_features + pairwise_offset, p25, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 26*batch_size*n_features + pairwise_offset, p26, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 27*batch_size*n_features + pairwise_offset, p27, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 28*batch_size*n_features + pairwise_offset, p28, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 29*batch_size*n_features + pairwise_offset, p29, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 30*batch_size*n_features + pairwise_offset, p30, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 31*batch_size*n_features + pairwise_offset, p31, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 32*batch_size*n_features + pairwise_offset, p32, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 33*batch_size*n_features + pairwise_offset, p33, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 34*batch_size*n_features + pairwise_offset, p34, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 35*batch_size*n_features + pairwise_offset, p35, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 36*batch_size*n_features + pairwise_offset, p36, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 37*batch_size*n_features + pairwise_offset, p37, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 38*batch_size*n_features + pairwise_offset, p38, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 39*batch_size*n_features + pairwise_offset, p39, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 40*batch_size*n_features + pairwise_offset, p40, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 41*batch_size*n_features + pairwise_offset, p41, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 42*batch_size*n_features + pairwise_offset, p42, mask=batch_feature_mask)
    tl.store(pairwise_ptr + 43*batch_size*n_features + pairwise_offset, p43, mask=batch_feature_mask)


@triton.jit
def assemble_kernel(
    transformed_ptr,
    pnorm_ptr,
    output_ptr,
    NORMALIZE: tl.constexpr,
    batch_size: tl.constexpr,
    n_features: tl.constexpr,
    BATCH_BLOCK: tl.constexpr,
    FEATURE_BLOCK: tl.constexpr,
    MV_DIM: tl.constexpr,
):
    """
    Gather linearly transformed pairwise products and compute the geometric product.
    """
    batch_block_id = tl.program_id(axis=0)
    thread_block_id = tl.program_id(axis=1)

    batch_ids = batch_block_id*BATCH_BLOCK + tl.arange(0, BATCH_BLOCK)
    feature_ids = thread_block_id*FEATURE_BLOCK + tl.arange(0, FEATURE_BLOCK)

    batch_mask = batch_ids < batch_size
    feature_mask = feature_ids < n_features
    batch_feature_mask = batch_mask[:, None] & feature_mask[None, :]

    feature_offset = batch_ids[:, None] * n_features * MV_DIM + feature_ids[None, :] * MV_DIM
    transformed_offset = batch_ids[:, None] * n_features + feature_ids[None, :]

    t0 = tl.load(transformed_ptr + 0 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t1 = tl.load(transformed_ptr + 1 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t2 = tl.load(transformed_ptr + 2 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t3 = tl.load(transformed_ptr + 3 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t4 = tl.load(transformed_ptr + 4 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t5 = tl.load(transformed_ptr + 5 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t6 = tl.load(transformed_ptr + 6 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t7 = tl.load(transformed_ptr + 7 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t8 = tl.load(transformed_ptr + 8 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t9 = tl.load(transformed_ptr + 9 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t10 = tl.load(transformed_ptr + 10 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t11 = tl.load(transformed_ptr + 11 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t12 = tl.load(transformed_ptr + 12 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t13 = tl.load(transformed_ptr + 13 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t14 = tl.load(transformed_ptr + 14 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t15 = tl.load(transformed_ptr + 15 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t16 = tl.load(transformed_ptr + 16 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t17 = tl.load(transformed_ptr + 17 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t18 = tl.load(transformed_ptr + 18 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t19 = tl.load(transformed_ptr + 19 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t20 = tl.load(transformed_ptr + 20 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t21 = tl.load(transformed_ptr + 21 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t22 = tl.load(transformed_ptr + 22 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t23 = tl.load(transformed_ptr + 23 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t24 = tl.load(transformed_ptr + 24 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t25 = tl.load(transformed_ptr + 25 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t26 = tl.load(transformed_ptr + 26 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t27 = tl.load(transformed_ptr + 27 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t28 = tl.load(transformed_ptr + 28 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t29 = tl.load(transformed_ptr + 29 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t30 = tl.load(transformed_ptr + 30 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t31 = tl.load(transformed_ptr + 31 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t32 = tl.load(transformed_ptr + 32 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t33 = tl.load(transformed_ptr + 33 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t34 = tl.load(transformed_ptr + 34 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t35 = tl.load(transformed_ptr + 35 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t36 = tl.load(transformed_ptr + 36 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t37 = tl.load(transformed_ptr + 37 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t38 = tl.load(transformed_ptr + 38 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t39 = tl.load(transformed_ptr + 39 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t40 = tl.load(transformed_ptr + 40 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t41 = tl.load(transformed_ptr + 41 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t42 = tl.load(transformed_ptr + 42 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)
    t43 = tl.load(transformed_ptr + 43 * batch_size * n_features + transformed_offset, mask=batch_feature_mask)

    o0 = t0 + t8 + t22 + t36
    o1 = t1 + t9 + t12 + t23 + t26 + t37
    o2 = t2 + t10 + t13 + t24 + t27 + t38
    o3 = t3 + t11 + t14 + t25 + t28 + t39
    o4 = t4 + t15 + t18 + t29 + t32 + t40
    o5 = t5 + t16 + t19 + t30 + t33 + t41
    o6 = t6 + t17 + t20 + t31 + t34 + t42
    o7 = t7 + t21 + t35 + t43
    
    if NORMALIZE:
        pn_scalar = tl.sum(o0 * o0, axis=1) / n_features
        pn_vector = tl.sum(o1 * o1 + o2 * o2 + o3 * o3, axis=1) / n_features
        pn_bivect = tl.sum(o4 * o4 + o5 * o5 + o6 * o6, axis=1) / n_features
        pn_pseudo = tl.sum(o7 * o7, axis=1) / n_features

        tl.atomic_add(pnorm_ptr + batch_ids * MV_DIM + 0, pn_scalar, mask=batch_mask)
        tl.atomic_add(pnorm_ptr + batch_ids * MV_DIM + 1, pn_vector, mask=batch_mask)
        tl.atomic_add(pnorm_ptr + batch_ids * MV_DIM + 2, pn_vector, mask=batch_mask)
        tl.atomic_add(pnorm_ptr + batch_ids * MV_DIM + 3, pn_vector, mask=batch_mask)
        tl.atomic_add(pnorm_ptr + batch_ids * MV_DIM + 4, pn_bivect, mask=batch_mask)
        tl.atomic_add(pnorm_ptr + batch_ids * MV_DIM + 5, pn_bivect, mask=batch_mask)
        tl.atomic_add(pnorm_ptr + batch_ids * MV_DIM + 6, pn_bivect, mask=batch_mask)
        tl.atomic_add(pnorm_ptr + batch_ids * MV_DIM + 7, pn_pseudo, mask=batch_mask)
        
    tl.store(output_ptr + feature_offset + 0, o0, mask=batch_feature_mask)
    tl.store(output_ptr + feature_offset + 1, o1, mask=batch_feature_mask)
    tl.store(output_ptr + feature_offset + 2, o2, mask=batch_feature_mask)
    tl.store(output_ptr + feature_offset + 3, o3, mask=batch_feature_mask)
    tl.store(output_ptr + feature_offset + 4, o4, mask=batch_feature_mask)
    tl.store(output_ptr + feature_offset + 5, o5, mask=batch_feature_mask)
    tl.store(output_ptr + feature_offset + 6, o6, mask=batch_feature_mask)
    tl.store(output_ptr + feature_offset + 7, o7, mask=batch_feature_mask)


@triton.jit
def normalize_with_sqrt_kernel(
    output_ptr,
    pnorm_ptr,
    batch_size: tl.constexpr,
    n_features: tl.constexpr,
    BATCH_BLOCK: tl.constexpr,
    FEATURE_BLOCK: tl.constexpr,
    MV_DIM: tl.constexpr,
    EPS: tl.constexpr,
):
    """Normalize the output by dividing each grade with root of its accumulated norm."""
    batch_block_id = tl.program_id(axis=0)
    thread_block_id = tl.program_id(axis=1)

    batch_ids = batch_block_id*BATCH_BLOCK + tl.arange(0, BATCH_BLOCK)
    feature_ids = thread_block_id*FEATURE_BLOCK + tl.arange(0, FEATURE_BLOCK)

    batch_mask = batch_ids < batch_size
    feature_mask = feature_ids < n_features
    batch_feature_mask = batch_mask[:, None, None] & feature_mask[None, :, None]

    component_ids = tl.arange(0, MV_DIM)[None, None, :]

    feature_offset = (batch_ids[:, None, None] * n_features * MV_DIM + feature_ids[None, :, None] * MV_DIM + component_ids)
    norm_indices = batch_ids[:, None, None] * MV_DIM + component_ids

    pnorm = tl.load(pnorm_ptr + norm_indices, mask=batch_mask[:, None, None])
    mv = tl.load(output_ptr + feature_offset, mask=batch_feature_mask)

    norm = tl.sqrt(pnorm + EPS)
    mv_normalized = mv / norm

    tl.store(output_ptr + feature_offset, mv_normalized, mask=batch_feature_mask)


def gelu_fc_geometric_product_norm_fwd(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    expansion_indices: torch.Tensor,
    normalize: bool,
    batch_block: int,
    feature_block: int,
    num_warps: int,
) -> torch.Tensor:
    """Fused operation: GELU non-linearity, fully connected geometric product, and grade-wise RMSNorm."""
    assert x.shape == y.shape
    assert x.shape[1] == weight.shape[1] == weight.shape[2]
    assert x.shape[2] == MV_DIM
    assert weight.shape[0] == NUM_PRODUCT_WEIGHTS

    B, N, _ = x.shape

    BATCH_BLOCK = min(batch_block, B)
    FEATURE_BLOCK = min(feature_block, N)

    num_blocks_batch = triton.cdiv(B, BATCH_BLOCK)
    num_blocks_features = triton.cdiv(N, FEATURE_BLOCK)

    pairwise = torch.empty((len(WEIGHT_EXPANSION), B, N), device=x.device, dtype=x.dtype)
    partial_norm = (torch.zeros((B, MV_DIM), device=x.device, dtype=x.dtype) if normalize else torch.zeros((1,), device=x.device, dtype=x.dtype))
    output = torch.empty((B, N, MV_DIM), device=x.device, dtype=x.dtype)

    grid = (num_blocks_batch, num_blocks_features)

    gelu_pairwise_kernel_fwd[grid](
        x,
        y,
        pairwise,
        B,
        N,
        BATCH_BLOCK,
        FEATURE_BLOCK,
        MV_DIM,
        num_warps=num_warps,
    )

    transformed = torch.bmm(pairwise, weight[expansion_indices])

    assemble_kernel[grid](
        transformed,
        partial_norm,
        output,
        normalize,
        B,
        N,
        BATCH_BLOCK,
        FEATURE_BLOCK,
        MV_DIM,
        num_warps=num_warps,
    )

    if normalize:
        normalize_with_sqrt_kernel[grid](
            output,
            partial_norm,
            B,
            N,
            BATCH_BLOCK,
            FEATURE_BLOCK,
            MV_DIM,
            EPS,
            num_warps=num_warps,
        )

    return output, pairwise, partial_norm


@triton.jit
def grad_o_dot_o_kernel(
    dot_ptr,
    pnorm_ptr,
    output_ptr,
    grad_output_ptr,
    batch_size: tl.constexpr,
    n_features: tl.constexpr,
    BATCH_BLOCK: tl.constexpr,
    FEATURE_BLOCK: tl.constexpr,
    MV_DIM: tl.constexpr,
    NUM_GRADES: tl.constexpr,
    EPS: tl.constexpr,
):
    """Compute the dot product of grad_output and output for each grade, accumulate across all features."""
    batch_block_id = tl.program_id(axis=0)
    thread_block_id = tl.program_id(axis=1)

    batch_ids = batch_block_id*BATCH_BLOCK + tl.arange(0, BATCH_BLOCK)
    feature_ids = thread_block_id*FEATURE_BLOCK + tl.arange(0, FEATURE_BLOCK)

    batch_mask = batch_ids < batch_size
    feature_mask = feature_ids < n_features
    batch_feature_mask = batch_mask[:, None] & feature_mask[None, :]

    offset = batch_ids[:, None] * n_features * MV_DIM + feature_ids[None, :] * MV_DIM

    go0 = tl.load(grad_output_ptr + offset + 0, mask=batch_feature_mask)
    go1 = tl.load(grad_output_ptr + offset + 1, mask=batch_feature_mask)
    go2 = tl.load(grad_output_ptr + offset + 2, mask=batch_feature_mask)
    go3 = tl.load(grad_output_ptr + offset + 3, mask=batch_feature_mask)
    go4 = tl.load(grad_output_ptr + offset + 4, mask=batch_feature_mask)
    go5 = tl.load(grad_output_ptr + offset + 5, mask=batch_feature_mask)
    go6 = tl.load(grad_output_ptr + offset + 6, mask=batch_feature_mask)
    go7 = tl.load(grad_output_ptr + offset + 7, mask=batch_feature_mask)

    pn_scalar = tl.load(pnorm_ptr + batch_ids*MV_DIM + 0, mask=batch_mask)[:, None]
    pn_vector = tl.load(pnorm_ptr + batch_ids*MV_DIM + 1, mask=batch_mask)[:, None]
    pn_bivect = tl.load(pnorm_ptr + batch_ids*MV_DIM + 4, mask=batch_mask)[:, None]
    pn_pseudo = tl.load(pnorm_ptr + batch_ids*MV_DIM + 7, mask=batch_mask)[:, None]

    o0 = tl.load(output_ptr + offset + 0, mask=batch_feature_mask)
    o1 = tl.load(output_ptr + offset + 1, mask=batch_feature_mask)
    o2 = tl.load(output_ptr + offset + 2, mask=batch_feature_mask)
    o3 = tl.load(output_ptr + offset + 3, mask=batch_feature_mask)
    o4 = tl.load(output_ptr + offset + 4, mask=batch_feature_mask)
    o5 = tl.load(output_ptr + offset + 5, mask=batch_feature_mask)
    o6 = tl.load(output_ptr + offset + 6, mask=batch_feature_mask)
    o7 = tl.load(output_ptr + offset + 7, mask=batch_feature_mask)

    rms_scalar = tl.sqrt(pn_scalar + EPS)
    rms_vector = tl.sqrt(pn_vector + EPS)
    rms_bivect = tl.sqrt(pn_bivect + EPS)
    rms_pseudo = tl.sqrt(pn_pseudo + EPS)

    dot_scalar = tl.sum(rms_scalar * go0 * o0, axis=1)
    dot_vector = tl.sum(rms_vector * (go1*o1 + go2*o2 + go3*o3), axis=1)
    dot_bivect = tl.sum(rms_bivect * (go4*o4 + go5*o5 + go6*o6), axis=1)
    dot_pseudo = tl.sum(rms_pseudo * go7 * o7, axis=1)

    tl.atomic_add(dot_ptr + batch_ids*NUM_GRADES + 0, dot_scalar, mask=batch_mask)
    tl.atomic_add(dot_ptr + batch_ids*NUM_GRADES + 1, dot_vector, mask=batch_mask)
    tl.atomic_add(dot_ptr + batch_ids*NUM_GRADES + 2, dot_bivect, mask=batch_mask)
    tl.atomic_add(dot_ptr + batch_ids*NUM_GRADES + 3, dot_pseudo, mask=batch_mask)
    
    
@triton.jit
def disassemble_kernel(
    grad_output_ptr,
    output_ptr,
    dot_ptr,
    grad_transformed_ptr,
    pnorm_ptr,
    NORMALIZE: tl.constexpr,
    batch_size: tl.constexpr,
    n_features: tl.constexpr,
    BATCH_BLOCK: tl.constexpr,
    FEATURE_BLOCK: tl.constexpr,
    MV_DIM: tl.constexpr,
    NUM_GRADES: tl.constexpr,
    EPS: tl.constexpr,
):
    """
    Gather linearly transformed pairwise products and compute the geometric product.
    """
    batch_block_id = tl.program_id(axis=0)
    thread_block_id = tl.program_id(axis=1)

    batch_ids = batch_block_id*BATCH_BLOCK + tl.arange(0, BATCH_BLOCK)
    feature_ids = thread_block_id*FEATURE_BLOCK + tl.arange(0, FEATURE_BLOCK)

    batch_mask = batch_ids < batch_size
    feature_mask = feature_ids < n_features
    batch_feature_mask = batch_mask[:, None] & feature_mask[None, :]

    feature_offset = batch_ids[:, None] * n_features * MV_DIM + feature_ids[None, :] * MV_DIM
    transformed_offset = batch_ids[:, None] * n_features + feature_ids[None, :]

    go0 = tl.load(grad_output_ptr + feature_offset + 0, mask=batch_feature_mask)
    go1 = tl.load(grad_output_ptr + feature_offset + 1, mask=batch_feature_mask)
    go2 = tl.load(grad_output_ptr + feature_offset + 2, mask=batch_feature_mask)
    go3 = tl.load(grad_output_ptr + feature_offset + 3, mask=batch_feature_mask)
    go4 = tl.load(grad_output_ptr + feature_offset + 4, mask=batch_feature_mask)
    go5 = tl.load(grad_output_ptr + feature_offset + 5, mask=batch_feature_mask)
    go6 = tl.load(grad_output_ptr + feature_offset + 6, mask=batch_feature_mask)
    go7 = tl.load(grad_output_ptr + feature_offset + 7, mask=batch_feature_mask)
    
    if NORMALIZE:
        o0 = tl.load(output_ptr + feature_offset + 0, mask=batch_feature_mask)
        o1 = tl.load(output_ptr + feature_offset + 1, mask=batch_feature_mask)
        o2 = tl.load(output_ptr + feature_offset + 2, mask=batch_feature_mask)
        o3 = tl.load(output_ptr + feature_offset + 3, mask=batch_feature_mask)
        o4 = tl.load(output_ptr + feature_offset + 4, mask=batch_feature_mask)
        o5 = tl.load(output_ptr + feature_offset + 5, mask=batch_feature_mask)
        o6 = tl.load(output_ptr + feature_offset + 6, mask=batch_feature_mask)
        o7 = tl.load(output_ptr + feature_offset + 7, mask=batch_feature_mask)
        
        pn_scalar = tl.load(pnorm_ptr + batch_ids * MV_DIM + 0, mask=batch_mask)[:, None]
        pn_vector = tl.load(pnorm_ptr + batch_ids * MV_DIM + 1, mask=batch_mask)[:, None]
        pn_bivect = tl.load(pnorm_ptr + batch_ids * MV_DIM + 4, mask=batch_mask)[:, None]
        pn_pseudo = tl.load(pnorm_ptr + batch_ids * MV_DIM + 7, mask=batch_mask)[:, None]
        
        dot_scalar = tl.load(dot_ptr + batch_ids * NUM_GRADES + 0, mask=batch_mask)[:, None]
        dot_vector = tl.load(dot_ptr + batch_ids * NUM_GRADES + 1, mask=batch_mask)[:, None]
        dot_bivect = tl.load(dot_ptr + batch_ids * NUM_GRADES + 2, mask=batch_mask)[:, None]
        dot_pseudo = tl.load(dot_ptr + batch_ids * NUM_GRADES + 3, mask=batch_mask)[:, None]
        
        rms_scalar = tl.sqrt(pn_scalar + EPS)
        rms_vector = tl.sqrt(pn_vector + EPS)
        rms_bivect = tl.sqrt(pn_bivect + EPS)
        rms_pseudo = tl.sqrt(pn_pseudo + EPS)
    
        go0 = go0 / rms_scalar - o0 * dot_scalar / (n_features * rms_scalar * rms_scalar)
        go1 = go1 / rms_vector - o1 * dot_vector / (n_features * rms_vector * rms_vector)
        go2 = go2 / rms_vector - o2 * dot_vector / (n_features * rms_vector * rms_vector)
        go3 = go3 / rms_vector - o3 * dot_vector / (n_features * rms_vector * rms_vector)
        go4 = go4 / rms_bivect - o4 * dot_bivect / (n_features * rms_bivect * rms_bivect)
        go5 = go5 / rms_bivect - o5 * dot_bivect / (n_features * rms_bivect * rms_bivect)
        go6 = go6 / rms_bivect - o6 * dot_bivect / (n_features * rms_bivect * rms_bivect)
        go7 = go7 / rms_pseudo - o7 * dot_pseudo / (n_features * rms_pseudo * rms_pseudo)
        
    tl.store(grad_transformed_ptr + 0 * batch_size * n_features + transformed_offset, go0, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 1 * batch_size * n_features + transformed_offset, go1, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 2 * batch_size * n_features + transformed_offset, go2, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 3 * batch_size * n_features + transformed_offset, go3, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 4 * batch_size * n_features + transformed_offset, go4, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 5 * batch_size * n_features + transformed_offset, go5, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 6 * batch_size * n_features + transformed_offset, go6, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 7 * batch_size * n_features + transformed_offset, go7, mask=batch_feature_mask)    
    tl.store(grad_transformed_ptr + 8 * batch_size * n_features + transformed_offset, go0, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 9 * batch_size * n_features + transformed_offset, go1, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 10 * batch_size * n_features + transformed_offset, go2, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 11 * batch_size * n_features + transformed_offset, go3, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 12 * batch_size * n_features + transformed_offset, go1, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 13 * batch_size * n_features + transformed_offset, go2, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 14 * batch_size * n_features + transformed_offset, go3, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 15 * batch_size * n_features + transformed_offset, go4, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 16 * batch_size * n_features + transformed_offset, go5, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 17 * batch_size * n_features + transformed_offset, go6, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 18 * batch_size * n_features + transformed_offset, go4, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 19 * batch_size * n_features + transformed_offset, go5, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 20 * batch_size * n_features + transformed_offset, go6, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 21 * batch_size * n_features + transformed_offset, go7, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 22 * batch_size * n_features + transformed_offset, go0, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 23 * batch_size * n_features + transformed_offset, go1, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 24 * batch_size * n_features + transformed_offset, go2, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 25 * batch_size * n_features + transformed_offset, go3, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 26 * batch_size * n_features + transformed_offset, go1, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 27 * batch_size * n_features + transformed_offset, go2, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 28 * batch_size * n_features + transformed_offset, go3, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 29 * batch_size * n_features + transformed_offset, go4, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 30 * batch_size * n_features + transformed_offset, go5, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 31 * batch_size * n_features + transformed_offset, go6, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 32 * batch_size * n_features + transformed_offset, go4, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 33 * batch_size * n_features + transformed_offset, go5, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 34 * batch_size * n_features + transformed_offset, go6, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 35 * batch_size * n_features + transformed_offset, go7, mask=batch_feature_mask)    
    tl.store(grad_transformed_ptr + 36 * batch_size * n_features + transformed_offset, go0, mask=batch_feature_mask)   
    tl.store(grad_transformed_ptr + 37 * batch_size * n_features + transformed_offset, go1, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 38 * batch_size * n_features + transformed_offset, go2, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 39 * batch_size * n_features + transformed_offset, go3, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 40 * batch_size * n_features + transformed_offset, go4, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 41 * batch_size * n_features + transformed_offset, go5, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 42 * batch_size * n_features + transformed_offset, go6, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 43 * batch_size * n_features + transformed_offset, go7, mask=batch_feature_mask)
    
    
@triton.jit
def gelu_pairwise_kernel_bwd(
    x_ptr,
    y_ptr,
    grad_pairwise_ptr,
    grad_x_ptr,
    grad_y_ptr,
    batch_size: tl.constexpr,
    n_features: tl.constexpr,
    BATCH_BLOCK: tl.constexpr,
    FEATURE_BLOCK: tl.constexpr,
    MV_DIM: tl.constexpr,
):
    """
    Apply GELU non-linearity to inputs and compute required pairwise products.
    """
    batch_block_id = tl.program_id(axis=0)
    thread_block_id = tl.program_id(axis=1)

    batch_ids = batch_block_id*BATCH_BLOCK + tl.arange(0, BATCH_BLOCK)
    feature_ids = thread_block_id*FEATURE_BLOCK + tl.arange(0, FEATURE_BLOCK)

    batch_mask = batch_ids < batch_size
    feature_mask = feature_ids < n_features
    batch_feature_mask = batch_mask[:, None] & feature_mask[None, :]

    feature_offset = batch_ids[:, None] * n_features * MV_DIM + feature_ids[None, :] * MV_DIM
    pairwise_offset = batch_ids[:, None] * n_features + feature_ids[None, :]

    gp0 = tl.load(grad_pairwise_ptr + 0 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp1 = tl.load(grad_pairwise_ptr + 1 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp2 = tl.load(grad_pairwise_ptr + 2 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp3 = tl.load(grad_pairwise_ptr + 3 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp4 = tl.load(grad_pairwise_ptr + 4 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp5 = tl.load(grad_pairwise_ptr + 5 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp6 = tl.load(grad_pairwise_ptr + 6 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp7 = tl.load(grad_pairwise_ptr + 7 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp8 = tl.load(grad_pairwise_ptr + 8 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp9 = tl.load(grad_pairwise_ptr + 9 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp10 = tl.load(grad_pairwise_ptr + 10 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp11 = tl.load(grad_pairwise_ptr + 11 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp12 = tl.load(grad_pairwise_ptr + 12 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp13 = tl.load(grad_pairwise_ptr + 13 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp14 = tl.load(grad_pairwise_ptr + 14 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp15 = tl.load(grad_pairwise_ptr + 15 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp16 = tl.load(grad_pairwise_ptr + 16 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp17 = tl.load(grad_pairwise_ptr + 17 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp18 = tl.load(grad_pairwise_ptr + 18 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp19 = tl.load(grad_pairwise_ptr + 19 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp20 = tl.load(grad_pairwise_ptr + 20 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp21 = tl.load(grad_pairwise_ptr + 21 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp22 = tl.load(grad_pairwise_ptr + 22 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp23 = tl.load(grad_pairwise_ptr + 23 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp24 = tl.load(grad_pairwise_ptr + 24 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp25 = tl.load(grad_pairwise_ptr + 25 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp26 = tl.load(grad_pairwise_ptr + 26 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp27 = tl.load(grad_pairwise_ptr + 27 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp28 = tl.load(grad_pairwise_ptr + 28 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp29 = tl.load(grad_pairwise_ptr + 29 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp30 = tl.load(grad_pairwise_ptr + 30 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp31 = tl.load(grad_pairwise_ptr + 31 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp32 = tl.load(grad_pairwise_ptr + 32 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp33 = tl.load(grad_pairwise_ptr + 33 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp34 = tl.load(grad_pairwise_ptr + 34 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp35 = tl.load(grad_pairwise_ptr + 35 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp36 = tl.load(grad_pairwise_ptr + 36 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp37 = tl.load(grad_pairwise_ptr + 37 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp38 = tl.load(grad_pairwise_ptr + 38 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp39 = tl.load(grad_pairwise_ptr + 39 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp40 = tl.load(grad_pairwise_ptr + 40 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp41 = tl.load(grad_pairwise_ptr + 41 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp42 = tl.load(grad_pairwise_ptr + 42 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    gp43 = tl.load(grad_pairwise_ptr + 43 * batch_size * n_features + pairwise_offset, mask=batch_feature_mask)
    
    x0_raw = tl.load(x_ptr + feature_offset + 0, mask=batch_feature_mask)
    x1_raw = tl.load(x_ptr + feature_offset + 1, mask=batch_feature_mask)
    x2_raw = tl.load(x_ptr + feature_offset + 2, mask=batch_feature_mask)
    x3_raw = tl.load(x_ptr + feature_offset + 3, mask=batch_feature_mask)
    x4_raw = tl.load(x_ptr + feature_offset + 4, mask=batch_feature_mask)
    x5_raw = tl.load(x_ptr + feature_offset + 5, mask=batch_feature_mask)
    x6_raw = tl.load(x_ptr + feature_offset + 6, mask=batch_feature_mask)
    x7_raw = tl.load(x_ptr + feature_offset + 7, mask=batch_feature_mask)

    y0_raw = tl.load(y_ptr + feature_offset + 0, mask=batch_feature_mask)
    y1_raw = tl.load(y_ptr + feature_offset + 1, mask=batch_feature_mask)
    y2_raw = tl.load(y_ptr + feature_offset + 2, mask=batch_feature_mask)
    y3_raw = tl.load(y_ptr + feature_offset + 3, mask=batch_feature_mask)
    y4_raw = tl.load(y_ptr + feature_offset + 4, mask=batch_feature_mask)
    y5_raw = tl.load(y_ptr + feature_offset + 5, mask=batch_feature_mask)
    y6_raw = tl.load(y_ptr + feature_offset + 6, mask=batch_feature_mask)
    y7_raw = tl.load(y_ptr + feature_offset + 7, mask=batch_feature_mask)
    
    # collect gradients from pairwise products
    gate_x = compute_gelu_gate(x0_raw)
    gate_y = compute_gelu_gate(y0_raw)

    x0 = x0_raw * gate_x
    x1 = x1_raw * gate_x
    x2 = x2_raw * gate_x
    x3 = x3_raw * gate_x
    x4 = x4_raw * gate_x
    x5 = x5_raw * gate_x
    x6 = x6_raw * gate_x
    x7 = x7_raw * gate_x

    y0 = y0_raw * gate_y
    y1 = y1_raw * gate_y
    y2 = y2_raw * gate_y
    y3 = y3_raw * gate_y
    y4 = y4_raw * gate_y
    y5 = y5_raw * gate_y
    y6 = y6_raw * gate_y
    y7 = y7_raw * gate_y
    
    x_grad_0 = gp0*y0 + gp1*y1 + gp2*y2 + gp3*y3 + gp4*y4 + gp5*y5 + gp6*y6 + gp7*y7
    x_grad_1 = gp8*y1 + gp9*y0 + gp13*y4 + gp14*y5 + gp15*y2 + gp16*y3 + gp20*y7 + gp21*y6
    x_grad_2 = gp8*y2 + gp10*y0 - gp12*y4 + gp14*y6 - gp15*y1 + gp17*y3 - gp19*y7 - gp21*y5
    x_grad_3 = gp8*y3 + gp11*y0 - gp12*y5 - gp13*y6 - gp16*y1 - gp17*y2 + gp18*y7 + gp21*y4
    x_grad_4 = -gp22*y4 + gp23*y2 - gp24*y1 - gp28*y7 + gp29*y0 + gp33*y6 - gp34*y5 + gp35*y3
    x_grad_5 = -gp22*y5 + gp23*y3 - gp25*y1 + gp27*y7 + gp30*y0 - gp32*y6 + gp34*y4 - gp35*y2
    x_grad_6 = -gp22*y6 + gp24*y3 - gp25*y2 - gp26*y7 + gp31*y0 + gp32*y5 - gp33*y4 + gp35*y1
    x_grad_7 = -gp36*y7 - gp37*y6 + gp38*y5 - gp39*y4 + gp40*y3 - gp41*y2 + gp42*y1 + gp43*y0

    y_grad_0 = gp0*x0 + gp9*x1 + gp10*x2 + gp11*x3 + gp29*x4 + gp30*x5 + gp31*x6 + gp43*x7
    y_grad_1 = gp1*x0 + gp8*x1 - gp15*x2 - gp16*x3 - gp24*x4 - gp25*x5 + gp35*x6 + gp42*x7
    y_grad_2 = gp2*x0 + gp8*x2 + gp15*x1 - gp17*x3 + gp23*x4 - gp35*x5 - gp25*x6 - gp41*x7
    y_grad_3 = gp3*x0 + gp8*x3 + gp16*x1 + gp17*x2 + gp23*x5 + gp24*x6 + gp35*x4 + gp40*x7
    y_grad_4 = gp4*x0 + gp13*x1 - gp12*x2 + gp21*x3 - gp22*x4 + gp34*x5 - gp33*x6 - gp39*x7
    y_grad_5 = gp5*x0 + gp14*x1 - gp21*x2 - gp12*x3 - gp22*x5 - gp34*x4 + gp32*x6 + gp38*x7
    y_grad_6 = gp6*x0 - gp13*x3 + gp14*x2 + gp21*x1 - gp22*x6 - gp32*x5 + gp33*x4 - gp37*x7
    y_grad_7 = gp7*x0 + gp20*x1 - gp19*x2 + gp18*x3 - gp28*x4 + gp27*x5 - gp26*x6 - gp36*x7
    
    # GELU gate gradients
    dgate_x = compute_gelu_gate_grad(x0_raw)
    dgate_y = compute_gelu_gate_grad(y0_raw)

    x_grad_0 = (gate_x + x0_raw*dgate_x) * x_grad_0 + dgate_x * (x1_raw*x_grad_1 + x2_raw*x_grad_2 + x3_raw*x_grad_3 + 
                                                                 x4_raw*x_grad_4 + x5_raw*x_grad_5 + x6_raw*x_grad_6 + 
                                                                 x7_raw*x_grad_7)
    x_grad_1 = gate_x * x_grad_1
    x_grad_2 = gate_x * x_grad_2
    x_grad_3 = gate_x * x_grad_3
    x_grad_4 = gate_x * x_grad_4
    x_grad_5 = gate_x * x_grad_5
    x_grad_6 = gate_x * x_grad_6
    x_grad_7 = gate_x * x_grad_7

    y_grad_0 = (gate_y + y0_raw*dgate_y) * y_grad_0 + dgate_y * (y1_raw*y_grad_1 + y2_raw*y_grad_2 + y3_raw*y_grad_3 + 
                                                                 y4_raw*y_grad_4 + y5_raw*y_grad_5 + y6_raw*y_grad_6 + 
                                                                 y7_raw*y_grad_7)
    y_grad_1 = gate_y * y_grad_1
    y_grad_2 = gate_y * y_grad_2
    y_grad_3 = gate_y * y_grad_3
    y_grad_4 = gate_y * y_grad_4
    y_grad_5 = gate_y * y_grad_5
    y_grad_6 = gate_y * y_grad_6
    y_grad_7 = gate_y * y_grad_7
    
    tl.store(grad_x_ptr + feature_offset + 0, x_grad_0, mask=batch_feature_mask)
    tl.store(grad_x_ptr + feature_offset + 1, x_grad_1, mask=batch_feature_mask)
    tl.store(grad_x_ptr + feature_offset + 2, x_grad_2, mask=batch_feature_mask)
    tl.store(grad_x_ptr + feature_offset + 3, x_grad_3, mask=batch_feature_mask)
    tl.store(grad_x_ptr + feature_offset + 4, x_grad_4, mask=batch_feature_mask)
    tl.store(grad_x_ptr + feature_offset + 5, x_grad_5, mask=batch_feature_mask)
    tl.store(grad_x_ptr + feature_offset + 6, x_grad_6, mask=batch_feature_mask)
    tl.store(grad_x_ptr + feature_offset + 7, x_grad_7, mask=batch_feature_mask)

    tl.store(grad_y_ptr + feature_offset + 0, y_grad_0, mask=batch_feature_mask)
    tl.store(grad_y_ptr + feature_offset + 1, y_grad_1, mask=batch_feature_mask)
    tl.store(grad_y_ptr + feature_offset + 2, y_grad_2, mask=batch_feature_mask)
    tl.store(grad_y_ptr + feature_offset + 3, y_grad_3, mask=batch_feature_mask)
    tl.store(grad_y_ptr + feature_offset + 4, y_grad_4, mask=batch_feature_mask)
    tl.store(grad_y_ptr + feature_offset + 5, y_grad_5, mask=batch_feature_mask)
    tl.store(grad_y_ptr + feature_offset + 6, y_grad_6, mask=batch_feature_mask)
    tl.store(grad_y_ptr + feature_offset + 7, y_grad_7, mask=batch_feature_mask)
    

def gelu_fc_geometric_product_norm_bwd(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    o: torch.Tensor,
    pairwise: torch.Tensor,
    partial_norm: torch.Tensor,
    grad_output: torch.Tensor,
    expansion_indices: torch.Tensor,
    normalize: bool,
    batch_block: int,
    feature_block: int,
    num_warps: int,
) -> torch.Tensor:
    """Backward pass for the fused operation."""
    B, N, _ = x.shape

    BATCH_BLOCK = min(batch_block, B)
    FEATURE_BLOCK = min(feature_block, N)

    num_blocks_batch = triton.cdiv(B, BATCH_BLOCK)
    num_blocks_features = triton.cdiv(N, FEATURE_BLOCK)

    grad_x = torch.zeros_like(x)
    grad_y = torch.zeros_like(y)
    dot = (torch.zeros((B, NUM_GRADES), device=x.device, dtype=x.dtype) if normalize else torch.empty(0))
    grad_weight = torch.zeros((NUM_PRODUCT_WEIGHTS, N, N), device=x.device, dtype=weight.dtype)
    grad_transformed = torch.empty((len(WEIGHT_EXPANSION), B, N), device=x.device, dtype=x.dtype)

    grid = (num_blocks_batch, num_blocks_features)

    if normalize:
        grad_o_dot_o_kernel[grid](
            dot,
            partial_norm,
            o,
            grad_output,
            B,
            N,
            BATCH_BLOCK,
            FEATURE_BLOCK,
            MV_DIM,
            NUM_GRADES,
            EPS,
            num_warps=num_warps,
        )

    disassemble_kernel[grid](
        grad_output,
        o,
        dot,
        grad_transformed,
        partial_norm,
        normalize,
        B,
        N,
        BATCH_BLOCK,
        FEATURE_BLOCK,
        MV_DIM,
        NUM_GRADES,
        EPS,
        num_warps=num_warps,
    )

    grad_pairwise = torch.bmm(grad_transformed, weight[expansion_indices].transpose(-2, -1))

    grad_weight.index_add_(0, expansion_indices, torch.bmm(pairwise.transpose(-2, -1), grad_transformed))

    gelu_pairwise_kernel_bwd[grid](
        x,
        y,
        grad_pairwise,
        grad_x,
        grad_y,
        B,
        N,
        BATCH_BLOCK,
        FEATURE_BLOCK,
        MV_DIM,
        num_warps=num_warps,
    )

    return grad_x, grad_y, grad_weight


class FullyConnectedGeluGeometricProductNorm3D(torch.autograd.Function):
    """
    Fused operation that applies GELU non-linearity to two multivector inputs,
    then computes their fully connected geometric product, and applies RMSNorm.

    Args:
        x (torch.Tensor): Input tensor of shape (B, N, MV_DIM).
        y (torch.Tensor): Input tensor of shape (B, N, MV_DIM).
        weight (torch.Tensor): Weight tensor of shape (N, NUM_PRODUCT_WEIGHTS), one weight per geometric product component.
        normalize (bool): Whether to apply RMSNorm after the geometric product.
        batch_block (int): Block size for batching in Triton kernel.
        feature_block (int): Block size for features in Triton kernel.
        num_warps (int): Number of warps to use in Triton kernel.

    Returns:
        torch.Tensor: Output tensor of shape (B, N, MV_DIM) after applying the fused operation.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, y, weight, normalize, batch_block=4, feature_block=128, num_warps=16):
        assert x.is_contiguous() and y.is_contiguous() and weight.is_contiguous()

        ctx.dtype = x.dtype
        ctx.batch_block = batch_block
        ctx.feature_block = feature_block
        ctx.num_warps = num_warps
        ctx.normalize = normalize

        expansion_indices = torch.tensor(WEIGHT_EXPANSION, device=x.device)

        o, pairwise, partial_norm = gelu_fc_geometric_product_norm_fwd(
            x,
            y,
            weight,
            expansion_indices,
            normalize,
            batch_block=batch_block,
            feature_block=feature_block,
            num_warps=num_warps,
        )

        ctx.save_for_backward(x, y, weight, o, pairwise, partial_norm, expansion_indices)

        return o.to(x.dtype)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        assert grad_output.is_contiguous()

        x, y, weight, o, pairwise, partial_norm, expansion_indices = ctx.saved_tensors

        grad_x, grad_y, grad_weight = gelu_fc_geometric_product_norm_bwd(
            x,
            y,
            weight,
            o,
            pairwise,
            partial_norm,
            grad_output,
            expansion_indices,
            ctx.normalize,
            batch_block=ctx.batch_block,
            feature_block=ctx.feature_block,
            num_warps=ctx.num_warps,
        )

        return grad_x, grad_y, grad_weight, None, None, None, None