import torch
import triton
import triton.language as tl

MV_DIM = 8
NUM_GRADES = 4
NUM_PRODUCT_WEIGHTS = 20
EPS = 1e-6

# tuned at RTX 4500
DEFAULT_BATCH_BLOCK = 4
DEFAULT_FEATURE_BLOCK = 128
DEFAULT_NUM_WARPS = 16
DEFAULT_NUM_STAGES = 1


@triton.jit
def compute_gelu_gate(x):
    """Compute the GELU gate Φ(x) := 0.5 * (1 + erf(x / sqrt(2)))"""
    return 0.5 * (1 + tl.erf(x.to(tl.float32) * 0.7071067811865475)).to(x.dtype)


@triton.jit
def compute_gelu_gate_grad(x):
    """Compute the gradient of the GELU gate = 1/sqrt(2pi) * exp(-x^2/2)"""
    return 0.3989422804 * tl.exp(-0.5 * x * x)


@triton.jit
def gelu_wgp_norm_kernel_fwd(
    x_ptr,
    y_ptr,
    output_ptr,
    weights_ptr,
    pnorm_ptr,
    NORMALIZE: tl.constexpr,
    batch_size: tl.constexpr,
    n_features: tl.constexpr,
    BATCH_BLOCK: tl.constexpr,
    FEATURE_BLOCK: tl.constexpr,
    NUM_PRODUCT_WEIGHTS: tl.constexpr,
):
    """
    Apply GELU non-linearity to inputs, compute weighted geometric product,
    and accumulate squared norms for grade-wise RMSNorm.
    """
    batch_block_id = tl.program_id(axis=0)
    thread_block_id = tl.program_id(axis=1)

    batch_ids = batch_block_id*BATCH_BLOCK + tl.arange(0, BATCH_BLOCK)
    feature_ids = thread_block_id*FEATURE_BLOCK + tl.arange(0, FEATURE_BLOCK)

    batch_mask = batch_ids < batch_size
    feature_mask = feature_ids < n_features
    batch_feature_mask = batch_mask[:, None] & feature_mask[None, :]

    stride_component = batch_size * n_features
    base_offset = batch_ids[:, None] * n_features + feature_ids[None, :]

    weight_offset = feature_ids * NUM_PRODUCT_WEIGHTS

    w0 = tl.load(weights_ptr + weight_offset + 0, mask=feature_mask)
    w1 = tl.load(weights_ptr + weight_offset + 1, mask=feature_mask)
    w2 = tl.load(weights_ptr + weight_offset + 2, mask=feature_mask)
    w3 = tl.load(weights_ptr + weight_offset + 3, mask=feature_mask)
    w4 = tl.load(weights_ptr + weight_offset + 4, mask=feature_mask)
    w5 = tl.load(weights_ptr + weight_offset + 5, mask=feature_mask)
    w6 = tl.load(weights_ptr + weight_offset + 6, mask=feature_mask)
    w7 = tl.load(weights_ptr + weight_offset + 7, mask=feature_mask)
    w8 = tl.load(weights_ptr + weight_offset + 8, mask=feature_mask)
    w9 = tl.load(weights_ptr + weight_offset + 9, mask=feature_mask)
    w10 = tl.load(weights_ptr + weight_offset + 10, mask=feature_mask)
    w11 = tl.load(weights_ptr + weight_offset + 11, mask=feature_mask)
    w12 = tl.load(weights_ptr + weight_offset + 12, mask=feature_mask)
    w13 = tl.load(weights_ptr + weight_offset + 13, mask=feature_mask)
    w14 = tl.load(weights_ptr + weight_offset + 14, mask=feature_mask)
    w15 = tl.load(weights_ptr + weight_offset + 15, mask=feature_mask)
    w16 = tl.load(weights_ptr + weight_offset + 16, mask=feature_mask)
    w17 = tl.load(weights_ptr + weight_offset + 17, mask=feature_mask)
    w18 = tl.load(weights_ptr + weight_offset + 18, mask=feature_mask)
    w19 = tl.load(weights_ptr + weight_offset + 19, mask=feature_mask)

    x0 = tl.load(x_ptr + 0 * stride_component + base_offset, mask=batch_feature_mask)
    x1 = tl.load(x_ptr + 1 * stride_component + base_offset, mask=batch_feature_mask)
    x2 = tl.load(x_ptr + 2 * stride_component + base_offset, mask=batch_feature_mask)
    x3 = tl.load(x_ptr + 3 * stride_component + base_offset, mask=batch_feature_mask)
    x4 = tl.load(x_ptr + 4 * stride_component + base_offset, mask=batch_feature_mask)
    x5 = tl.load(x_ptr + 5 * stride_component + base_offset, mask=batch_feature_mask)
    x6 = tl.load(x_ptr + 6 * stride_component + base_offset, mask=batch_feature_mask)
    x7 = tl.load(x_ptr + 7 * stride_component + base_offset, mask=batch_feature_mask)

    y0 = tl.load(y_ptr + 0 * stride_component + base_offset, mask=batch_feature_mask)
    y1 = tl.load(y_ptr + 1 * stride_component + base_offset, mask=batch_feature_mask)
    y2 = tl.load(y_ptr + 2 * stride_component + base_offset, mask=batch_feature_mask)
    y3 = tl.load(y_ptr + 3 * stride_component + base_offset, mask=batch_feature_mask)
    y4 = tl.load(y_ptr + 4 * stride_component + base_offset, mask=batch_feature_mask)
    y5 = tl.load(y_ptr + 5 * stride_component + base_offset, mask=batch_feature_mask)
    y6 = tl.load(y_ptr + 6 * stride_component + base_offset, mask=batch_feature_mask)
    y7 = tl.load(y_ptr + 7 * stride_component + base_offset, mask=batch_feature_mask)

    # Apply GELU gate
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

    # Compute geometric product
    o0 = (w0*x0*y0 + w4 * (x1*y1 + x2*y2 + x3*y3) - w10 * (x4*y4 + x5*y5 + x6*y6) - w16*x7*y7)
    o1 = (w1*x0*y1 + w5*x1*y0 - w6 * (x2*y4 + x3*y5) + w11 * (x4*y2 + x5*y3) - w12*x6*y7 - w17*x7*y6)
    o2 = (w1*x0*y2 + w6*x1*y4 + w5*x2*y0 - w6*x3*y6 - w11*x4*y1 + w12*x5*y7 + w11*x6*y3 + w17*x7*y5)
    o3 = (w1*x0*y3 + w6 * (x1*y5 + x2*y6) + w5*x3*y0 - w12*x4*y7 - w11 * (x5*y1 + x6*y2) - w17*x7*y4)
    o4 = (w2*x0*y4 + w7*x1*y2 - w7*x2*y1 + w8*x3*y7 + w13*x4*y0 - w14*x5*y6 + w14*x6*y5 + w18*x7*y3)
    o5 = (w2*x0*y5 + w7*x1*y3 - w8*x2*y7 - w7*x3*y1 + w14*x4*y6 + w13*x5*y0 - w14*x6*y4 - w18*x7*y2)
    o6 = (w2*x0*y6 + w8*x1*y7 + w7*x2*y3 - w7*x3*y2 - w14*x4*y5 + w14*x5*y4 + w13*x6*y0 + w18*x7*y1)
    o7 = (w3*x0*y7 + w9*x1*y6 - w9*x2*y5 + w9*x3*y4 + w15*x4*y3 - w15*x5*y2 + w15*x6*y1 + w19*x7*y0)

    if NORMALIZE:
        pn_scalar = tl.sum(o0 * o0, axis=1) / n_features
        pn_vector = tl.sum(o1*o1 + o2*o2 + o3*o3, axis=1) / n_features
        pn_bivect = tl.sum(o4*o4 + o5*o5 + o6*o6, axis=1) / n_features
        pn_pseudo = tl.sum(o7 * o7, axis=1) / n_features

        tl.atomic_add(pnorm_ptr + 0*batch_size + batch_ids, pn_scalar, mask=batch_mask)
        tl.atomic_add(pnorm_ptr + 1*batch_size + batch_ids, pn_vector, mask=batch_mask)
        tl.atomic_add(pnorm_ptr + 2*batch_size + batch_ids, pn_vector, mask=batch_mask)
        tl.atomic_add(pnorm_ptr + 3*batch_size + batch_ids, pn_vector, mask=batch_mask)
        tl.atomic_add(pnorm_ptr + 4*batch_size + batch_ids, pn_bivect, mask=batch_mask)
        tl.atomic_add(pnorm_ptr + 5*batch_size + batch_ids, pn_bivect, mask=batch_mask)
        tl.atomic_add(pnorm_ptr + 6*batch_size + batch_ids, pn_bivect, mask=batch_mask)
        tl.atomic_add(pnorm_ptr + 7*batch_size + batch_ids, pn_pseudo, mask=batch_mask)

    tl.store(output_ptr + 0 * stride_component + base_offset, o0, mask=batch_feature_mask)
    tl.store(output_ptr + 1 * stride_component + base_offset, o1, mask=batch_feature_mask)
    tl.store(output_ptr + 2 * stride_component + base_offset, o2, mask=batch_feature_mask)
    tl.store(output_ptr + 3 * stride_component + base_offset, o3, mask=batch_feature_mask)
    tl.store(output_ptr + 4 * stride_component + base_offset, o4, mask=batch_feature_mask)
    tl.store(output_ptr + 5 * stride_component + base_offset, o5, mask=batch_feature_mask)
    tl.store(output_ptr + 6 * stride_component + base_offset, o6, mask=batch_feature_mask)
    tl.store(output_ptr + 7 * stride_component + base_offset, o7, mask=batch_feature_mask)


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

    feature_offset = (component_ids * batch_size * n_features + 
                     batch_ids[:, None, None] * n_features + 
                     feature_ids[None, :, None])
    
    norm_indices = component_ids * batch_size + batch_ids[:, None, None]

    pnorm = tl.load(pnorm_ptr + norm_indices, mask=batch_mask[:, None, None])
    mv = tl.load(output_ptr + feature_offset, mask=batch_feature_mask)

    norm = tl.sqrt(pnorm + EPS)
    mv_normalized = mv / norm

    tl.store(output_ptr + feature_offset, mv_normalized, mask=batch_feature_mask)


def gelu_geometric_product_norm_fwd(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    normalize: bool,
) -> torch.Tensor:
    """Fused operation: GELU non-linearity, weighted geometric product, and grade-wise RMSNorm."""
    assert x.shape == y.shape
    assert x.shape[0] == MV_DIM
    assert x.shape[2] == weight.shape[0]
    assert weight.shape[1] == NUM_PRODUCT_WEIGHTS

    _, B, N = x.shape

    BATCH_BLOCK = min(DEFAULT_BATCH_BLOCK, B)
    FEATURE_BLOCK = min(DEFAULT_FEATURE_BLOCK, N)

    num_blocks_batch = triton.cdiv(B, BATCH_BLOCK)
    num_blocks_features = triton.cdiv(N, FEATURE_BLOCK)

    output = torch.empty_like(x)
    partial_norm = (torch.zeros((MV_DIM, B), device=x.device, dtype=x.dtype) if normalize 
                   else torch.zeros((1,), device=x.device, dtype=x.dtype))

    grid = (num_blocks_batch, num_blocks_features)

    gelu_wgp_norm_kernel_fwd[grid](
        x,
        y,
        output,
        weight,
        partial_norm,
        normalize,
        B,
        N,
        BATCH_BLOCK,
        FEATURE_BLOCK,
        NUM_PRODUCT_WEIGHTS,
        num_warps=DEFAULT_NUM_WARPS,
        num_stages=DEFAULT_NUM_STAGES,
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
            num_warps=DEFAULT_NUM_WARPS,
            num_stages=DEFAULT_NUM_STAGES,
        )

    return output, partial_norm


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

    stride_component = batch_size * n_features
    offset = batch_ids[:, None] * n_features + feature_ids[None, :]

    go0 = tl.load(grad_output_ptr + 0 * stride_component + offset, mask=batch_feature_mask)
    go1 = tl.load(grad_output_ptr + 1 * stride_component + offset, mask=batch_feature_mask)
    go2 = tl.load(grad_output_ptr + 2 * stride_component + offset, mask=batch_feature_mask)
    go3 = tl.load(grad_output_ptr + 3 * stride_component + offset, mask=batch_feature_mask)
    go4 = tl.load(grad_output_ptr + 4 * stride_component + offset, mask=batch_feature_mask)
    go5 = tl.load(grad_output_ptr + 5 * stride_component + offset, mask=batch_feature_mask)
    go6 = tl.load(grad_output_ptr + 6 * stride_component + offset, mask=batch_feature_mask)
    go7 = tl.load(grad_output_ptr + 7 * stride_component + offset, mask=batch_feature_mask)

    pn_scalar = tl.load(pnorm_ptr + 0*batch_size + batch_ids, mask=batch_mask)[:, None]
    pn_vector = tl.load(pnorm_ptr + 1*batch_size + batch_ids, mask=batch_mask)[:, None]
    pn_bivect = tl.load(pnorm_ptr + 4*batch_size + batch_ids, mask=batch_mask)[:, None]
    pn_pseudo = tl.load(pnorm_ptr + 7*batch_size + batch_ids, mask=batch_mask)[:, None]

    o0 = tl.load(output_ptr + 0 * stride_component + offset, mask=batch_feature_mask)
    o1 = tl.load(output_ptr + 1 * stride_component + offset, mask=batch_feature_mask)
    o2 = tl.load(output_ptr + 2 * stride_component + offset, mask=batch_feature_mask)
    o3 = tl.load(output_ptr + 3 * stride_component + offset, mask=batch_feature_mask)
    o4 = tl.load(output_ptr + 4 * stride_component + offset, mask=batch_feature_mask)
    o5 = tl.load(output_ptr + 5 * stride_component + offset, mask=batch_feature_mask)
    o6 = tl.load(output_ptr + 6 * stride_component + offset, mask=batch_feature_mask)
    o7 = tl.load(output_ptr + 7 * stride_component + offset, mask=batch_feature_mask)

    rms_scalar = tl.sqrt(pn_scalar + EPS)
    rms_vector = tl.sqrt(pn_vector + EPS)
    rms_bivect = tl.sqrt(pn_bivect + EPS)
    rms_pseudo = tl.sqrt(pn_pseudo + EPS)

    dot_scalar = tl.sum(rms_scalar * go0 * o0, axis=1)
    dot_vector = tl.sum(rms_vector * (go1*o1 + go2*o2 + go3*o3), axis=1)
    dot_bivect = tl.sum(rms_bivect * (go4*o4 + go5*o5 + go6*o6), axis=1)
    dot_pseudo = tl.sum(rms_pseudo * go7 * o7, axis=1)

    tl.atomic_add(dot_ptr + 0*batch_size + batch_ids, dot_scalar, mask=batch_mask)
    tl.atomic_add(dot_ptr + 1*batch_size + batch_ids, dot_vector, mask=batch_mask)
    tl.atomic_add(dot_ptr + 2*batch_size + batch_ids, dot_bivect, mask=batch_mask)
    tl.atomic_add(dot_ptr + 3*batch_size + batch_ids, dot_pseudo, mask=batch_mask)


@triton.jit
def gelu_wgp_norm_kernel_bwd(
    x_ptr,
    y_ptr,
    output_ptr,
    weights_ptr,
    dot_ptr,
    pnorm_ptr,
    grad_output_ptr,
    grad_x_ptr,
    grad_y_ptr,
    grad_weight_ptr,
    NORMALIZE: tl.constexpr,
    batch_size: tl.constexpr,
    n_features: tl.constexpr,
    BATCH_BLOCK: tl.constexpr,
    FEATURE_BLOCK: tl.constexpr,
    MV_DIM: tl.constexpr,
    NUM_GRADES: tl.constexpr,
    NUM_PRODUCT_WEIGHTS: tl.constexpr,
    EPS: tl.constexpr,
):
    """Compute gradients w.r.t. inputs and weights of the fused operation."""
    batch_block_id = tl.program_id(axis=0)
    thread_block_id = tl.program_id(axis=1)

    batch_ids = batch_block_id*BATCH_BLOCK + tl.arange(0, BATCH_BLOCK)
    feature_ids = thread_block_id*FEATURE_BLOCK + tl.arange(0, FEATURE_BLOCK)

    batch_mask = batch_ids < batch_size
    feature_mask = feature_ids < n_features
    batch_feature_mask = batch_mask[:, None] & feature_mask[None, :]

    stride_component = batch_size * n_features
    base_offset = batch_ids[:, None] * n_features + feature_ids[None, :]
    
    weight_offset = feature_ids * NUM_PRODUCT_WEIGHTS
    block_offset = batch_block_id * n_features * NUM_PRODUCT_WEIGHTS

    w0 = tl.load(weights_ptr + weight_offset + 0, mask=feature_mask)
    w1 = tl.load(weights_ptr + weight_offset + 1, mask=feature_mask)
    w2 = tl.load(weights_ptr + weight_offset + 2, mask=feature_mask)
    w3 = tl.load(weights_ptr + weight_offset + 3, mask=feature_mask)
    w4 = tl.load(weights_ptr + weight_offset + 4, mask=feature_mask)
    w5 = tl.load(weights_ptr + weight_offset + 5, mask=feature_mask)
    w6 = tl.load(weights_ptr + weight_offset + 6, mask=feature_mask)
    w7 = tl.load(weights_ptr + weight_offset + 7, mask=feature_mask)
    w8 = tl.load(weights_ptr + weight_offset + 8, mask=feature_mask)
    w9 = tl.load(weights_ptr + weight_offset + 9, mask=feature_mask)
    w10 = tl.load(weights_ptr + weight_offset + 10, mask=feature_mask)
    w11 = tl.load(weights_ptr + weight_offset + 11, mask=feature_mask)
    w12 = tl.load(weights_ptr + weight_offset + 12, mask=feature_mask)
    w13 = tl.load(weights_ptr + weight_offset + 13, mask=feature_mask)
    w14 = tl.load(weights_ptr + weight_offset + 14, mask=feature_mask)
    w15 = tl.load(weights_ptr + weight_offset + 15, mask=feature_mask)
    w16 = tl.load(weights_ptr + weight_offset + 16, mask=feature_mask)
    w17 = tl.load(weights_ptr + weight_offset + 17, mask=feature_mask)
    w18 = tl.load(weights_ptr + weight_offset + 18, mask=feature_mask)
    w19 = tl.load(weights_ptr + weight_offset + 19, mask=feature_mask)

    x0_raw = tl.load(x_ptr + 0 * stride_component + base_offset, mask=batch_feature_mask)
    x1_raw = tl.load(x_ptr + 1 * stride_component + base_offset, mask=batch_feature_mask)
    x2_raw = tl.load(x_ptr + 2 * stride_component + base_offset, mask=batch_feature_mask)
    x3_raw = tl.load(x_ptr + 3 * stride_component + base_offset, mask=batch_feature_mask)
    x4_raw = tl.load(x_ptr + 4 * stride_component + base_offset, mask=batch_feature_mask)
    x5_raw = tl.load(x_ptr + 5 * stride_component + base_offset, mask=batch_feature_mask)
    x6_raw = tl.load(x_ptr + 6 * stride_component + base_offset, mask=batch_feature_mask)
    x7_raw = tl.load(x_ptr + 7 * stride_component + base_offset, mask=batch_feature_mask)

    y0_raw = tl.load(y_ptr + 0 * stride_component + base_offset, mask=batch_feature_mask)
    y1_raw = tl.load(y_ptr + 1 * stride_component + base_offset, mask=batch_feature_mask)
    y2_raw = tl.load(y_ptr + 2 * stride_component + base_offset, mask=batch_feature_mask)
    y3_raw = tl.load(y_ptr + 3 * stride_component + base_offset, mask=batch_feature_mask)
    y4_raw = tl.load(y_ptr + 4 * stride_component + base_offset, mask=batch_feature_mask)
    y5_raw = tl.load(y_ptr + 5 * stride_component + base_offset, mask=batch_feature_mask)
    y6_raw = tl.load(y_ptr + 6 * stride_component + base_offset, mask=batch_feature_mask)
    y7_raw = tl.load(y_ptr + 7 * stride_component + base_offset, mask=batch_feature_mask)

    go0 = tl.load(grad_output_ptr + 0 * stride_component + base_offset, mask=batch_feature_mask)
    go1 = tl.load(grad_output_ptr + 1 * stride_component + base_offset, mask=batch_feature_mask)
    go2 = tl.load(grad_output_ptr + 2 * stride_component + base_offset, mask=batch_feature_mask)
    go3 = tl.load(grad_output_ptr + 3 * stride_component + base_offset, mask=batch_feature_mask)
    go4 = tl.load(grad_output_ptr + 4 * stride_component + base_offset, mask=batch_feature_mask)
    go5 = tl.load(grad_output_ptr + 5 * stride_component + base_offset, mask=batch_feature_mask)
    go6 = tl.load(grad_output_ptr + 6 * stride_component + base_offset, mask=batch_feature_mask)
    go7 = tl.load(grad_output_ptr + 7 * stride_component + base_offset, mask=batch_feature_mask)

    if NORMALIZE:
        o0 = tl.load(output_ptr + 0 * stride_component + base_offset, mask=batch_feature_mask)
        o1 = tl.load(output_ptr + 1 * stride_component + base_offset, mask=batch_feature_mask)
        o2 = tl.load(output_ptr + 2 * stride_component + base_offset, mask=batch_feature_mask)
        o3 = tl.load(output_ptr + 3 * stride_component + base_offset, mask=batch_feature_mask)
        o4 = tl.load(output_ptr + 4 * stride_component + base_offset, mask=batch_feature_mask)
        o5 = tl.load(output_ptr + 5 * stride_component + base_offset, mask=batch_feature_mask)
        o6 = tl.load(output_ptr + 6 * stride_component + base_offset, mask=batch_feature_mask)
        o7 = tl.load(output_ptr + 7 * stride_component + base_offset, mask=batch_feature_mask)

        pn_scalar = tl.load(pnorm_ptr + 0*batch_size + batch_ids, mask=batch_mask)[:, None]
        pn_vector = tl.load(pnorm_ptr + 1*batch_size + batch_ids, mask=batch_mask)[:, None]
        pn_bivect = tl.load(pnorm_ptr + 4*batch_size + batch_ids, mask=batch_mask)[:, None]
        pn_pseudo = tl.load(pnorm_ptr + 7*batch_size + batch_ids, mask=batch_mask)[:, None]

        dot_scalar = tl.load(dot_ptr + 0*batch_size + batch_ids, mask=batch_mask)[:, None]
        dot_vector = tl.load(dot_ptr + 1*batch_size + batch_ids, mask=batch_mask)[:, None]
        dot_bivect = tl.load(dot_ptr + 2*batch_size + batch_ids, mask=batch_mask)[:, None]
        dot_pseudo = tl.load(dot_ptr + 3*batch_size + batch_ids, mask=batch_mask)[:, None]

        rms_scalar = tl.sqrt(pn_scalar + EPS)
        rms_vector = tl.sqrt(pn_vector + EPS)
        rms_bivect = tl.sqrt(pn_bivect + EPS)
        rms_pseudo = tl.sqrt(pn_pseudo + EPS)

        go0 = go0/rms_scalar - o0 * dot_scalar / (n_features*rms_scalar*rms_scalar)
        go1 = go1/rms_vector - o1 * dot_vector / (n_features*rms_vector*rms_vector)
        go2 = go2/rms_vector - o2 * dot_vector / (n_features*rms_vector*rms_vector)
        go3 = go3/rms_vector - o3 * dot_vector / (n_features*rms_vector*rms_vector)
        go4 = go4/rms_bivect - o4 * dot_bivect / (n_features*rms_bivect*rms_bivect)
        go5 = go5/rms_bivect - o5 * dot_bivect / (n_features*rms_bivect*rms_bivect)
        go6 = go6/rms_bivect - o6 * dot_bivect / (n_features*rms_bivect*rms_bivect)
        go7 = go7/rms_pseudo - o7 * dot_pseudo / (n_features*rms_pseudo*rms_pseudo)

    # weighted geometric product backward
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

    tmp0 = go0 * w0
    tmp1 = go7 * w3
    tmp2 = go1 * y1
    tmp3 = go2 * y2
    tmp4 = go3 * y3
    tmp5 = go4 * y4
    tmp6 = go5 * y5
    tmp7 = go6 * y6
    tmp8 = go0 * w4
    tmp9 = w5 * y0
    tmp10 = w8 * y7
    tmp11 = go7 * w9
    tmp12 = go1 * w6
    tmp13 = w13 * y0
    tmp14 = go7 * w15
    tmp15 = go0 * w10
    tmp16 = w12 * y7
    tmp17 = go3 * w11
    tmp18 = go7 * w19
    tmp19 = go0 * w16
    tmp20 = go4 * y3
    tmp21 = go6 * y1
    tmp22 = go5 * y2
    tmp23 = go1 * y6
    tmp24 = go3 * y4
    tmp25 = go4 * x4
    tmp26 = go5 * x5
    tmp27 = go6 * x6
    tmp28 = go1 * x1
    tmp29 = go2 * x2
    tmp30 = go3 * x3
    tmp31 = w1 * x0
    tmp32 = w18 * x7
    tmp33 = w2 * x0
    tmp34 = w17 * x7
    tmp35 = go4 * x3
    tmp36 = go6 * x1
    tmp37 = go5 * x2
    tmp38 = go1 * x6
    tmp39 = go3 * x4

    x_grad_0 = (tmp0*y0 + tmp1*y7 + w1 * (tmp2+tmp3+tmp4) + w2 * (tmp5+tmp6+tmp7))
    x_grad_1 = (go1*tmp9 + go6*tmp10 + tmp11*y6 + tmp8*y1 + w6 * (go2*y4 + go3*y5) + w7 * (go4*y2 + go5*y3))
    x_grad_2 = (go2*tmp9 + go3*w6*y6 - go5*tmp10 - tmp11*y5 - tmp12*y4 + tmp8*y2 + w7 * (-go4 * y1 + go6*y3))
    x_grad_3 = (go3*tmp9 + go4*tmp10 + tmp11*y4 + tmp8*y3 - w6 * (go1*y5 + go2*y6) - w7 * (go5*y1 + go6*y2))
    x_grad_4 = (-go3 * tmp16 + go4*tmp13 + tmp14*y3 - tmp15*y4 + w11 * (go1*y2 - go2*y1) + w14 * (go5*y6 - go6*y5))
    x_grad_5 = (go1*w11*y3 + go2*w12*y7 - go4*w14*y6 + go5*w13*y0 + go6*w14*y4 - tmp14*y2 - tmp15*y5 - tmp17*y1)
    x_grad_6 = (-go1 * tmp16 + go6*tmp13 + tmp14*y1 - tmp15*y6 + w11 * (go2*y3 - go3*y2) + w14 * (go4*y5 - go5*y4))
    x_grad_7 = (tmp18*y0 - tmp19*y7 + w17 * (go2*y5 - tmp23 - tmp24) + w18 * (tmp20+tmp21-tmp22))

    y_grad_0 = (tmp0*x0 + tmp18*x7 + w13 * (tmp25+tmp26+tmp27) + w5 * (tmp28+tmp29+tmp30))
    y_grad_1 = (go1*tmp31 + go6*tmp32 + tmp14*x6 + tmp8*x1 - w11 * (go2*x4 + go3*x5) - w7 * (go4*x2 + go5*x3))
    y_grad_2 = (go1*w11*x4 + go2*tmp31 + go4*w7*x1 - go5*tmp32 - go6*w7*x3 - tmp14*x5 - tmp17*x6 + tmp8*x2)
    y_grad_3 = (go3*tmp31 + go4*tmp32 + tmp14*x4 + tmp8*x3 + w11 * (go1*x5 + go2*x6) + w7 * (go5*x1 + go6*x2))
    y_grad_4 = (-go3 * tmp34 + go4*tmp33 + tmp11*x3 - tmp15*x4 + w14 * (-go5 * x6 + go6*x5) + w6 * (-go1 * x2 + go2*x1))
    y_grad_5 = (go2*w17*x7 + go3*w6*x1 + go4*w14*x6 + go5*w2*x0 - go6*w14*x4 - tmp11*x2 - tmp12*x3 - tmp15*x5)
    y_grad_6 = (-go1 * tmp34 + go6*tmp33 + tmp11*x1 - tmp15*x6 + w14 * (-go4 * x5 + go5*x4) + w6 * (-go2 * x3 + go3*x2))
    y_grad_7 = (tmp1*x0 - tmp19*x7 + w12 * (go2*x5 - tmp38 - tmp39) + w8 * (tmp35+tmp36-tmp37))

    w_grad_0 = tl.sum(go0 * x0 * y0, axis=0)
    w_grad_1 = tl.sum(tmp2*x0 + tmp3*x0 + tmp4*x0, axis=0)
    w_grad_2 = tl.sum(tmp5*x0 + tmp6*x0 + tmp7*x0, axis=0)
    w_grad_3 = tl.sum(go7 * x0 * y7, axis=0)
    w_grad_4 = tl.sum(go0 * (x1*y1 + x2*y2 + x3*y3), axis=0)
    w_grad_5 = tl.sum(tmp28*y0 + tmp29*y0 + tmp30*y0, axis=0)
    w_grad_6 = tl.sum(go1 * (-x2 * y4 - x3*y5) + go2 * (x1*y4 - x3*y6) + go3 * (x1*y5 + x2*y6), axis=0)
    w_grad_7 = tl.sum(go4 * (x1*y2 - x2*y1) + go5 * (x1*y3 - x3*y1) + go6 * (x2*y3 - x3*y2), axis=0)
    w_grad_8 = tl.sum(tmp35*y7 + tmp36*y7 - tmp37*y7, axis=0)
    w_grad_9 = tl.sum(go7 * (x1*y6 - x2*y5 + x3*y4), axis=0)
    w_grad_10 = tl.sum(go0 * (-x4 * y4 - x5*y5 - x6*y6), axis=0)
    w_grad_11 = tl.sum(go1 * (x4*y2 + x5*y3) + go2 * (-x4 * y1 + x6*y3) + go3 * (-x5 * y1 - x6*y2), axis=0)
    w_grad_12 = tl.sum(go2*x5*y7 - tmp38*y7 - tmp39*y7, axis=0)
    w_grad_13 = tl.sum(tmp25*y0 + tmp26*y0 + tmp27*y0, axis=0)
    w_grad_14 = tl.sum(go4 * (-x5 * y6 + x6*y5) + go5 * (x4*y6 - x6*y4) + go6 * (-x4 * y5 + x5*y4), axis=0)
    w_grad_15 = tl.sum(go7 * (x4*y3 - x5*y2 + x6*y1), axis=0)
    w_grad_16 = tl.sum(-go0 * x7 * y7, axis=0)
    w_grad_17 = tl.sum(go2*x7*y5 - tmp23*x7 - tmp24*x7, axis=0)
    w_grad_18 = tl.sum(tmp20*x7 + tmp21*x7 - tmp22*x7, axis=0)
    w_grad_19 = tl.sum(go7 * x7 * y0, axis=0)

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

    tl.store(grad_x_ptr + 0 * stride_component + base_offset, x_grad_0, mask=batch_feature_mask)
    tl.store(grad_x_ptr + 1 * stride_component + base_offset, x_grad_1, mask=batch_feature_mask)
    tl.store(grad_x_ptr + 2 * stride_component + base_offset, x_grad_2, mask=batch_feature_mask)
    tl.store(grad_x_ptr + 3 * stride_component + base_offset, x_grad_3, mask=batch_feature_mask)
    tl.store(grad_x_ptr + 4 * stride_component + base_offset, x_grad_4, mask=batch_feature_mask)
    tl.store(grad_x_ptr + 5 * stride_component + base_offset, x_grad_5, mask=batch_feature_mask)
    tl.store(grad_x_ptr + 6 * stride_component + base_offset, x_grad_6, mask=batch_feature_mask)
    tl.store(grad_x_ptr + 7 * stride_component + base_offset, x_grad_7, mask=batch_feature_mask)

    tl.store(grad_y_ptr + 0 * stride_component + base_offset, y_grad_0, mask=batch_feature_mask)
    tl.store(grad_y_ptr + 1 * stride_component + base_offset, y_grad_1, mask=batch_feature_mask)
    tl.store(grad_y_ptr + 2 * stride_component + base_offset, y_grad_2, mask=batch_feature_mask)
    tl.store(grad_y_ptr + 3 * stride_component + base_offset, y_grad_3, mask=batch_feature_mask)
    tl.store(grad_y_ptr + 4 * stride_component + base_offset, y_grad_4, mask=batch_feature_mask)
    tl.store(grad_y_ptr + 5 * stride_component + base_offset, y_grad_5, mask=batch_feature_mask)
    tl.store(grad_y_ptr + 6 * stride_component + base_offset, y_grad_6, mask=batch_feature_mask)
    tl.store(grad_y_ptr + 7 * stride_component + base_offset, y_grad_7, mask=batch_feature_mask)

    tl.store(grad_weight_ptr + block_offset + weight_offset + 0, w_grad_0, mask=feature_mask)
    tl.store(grad_weight_ptr + block_offset + weight_offset + 1, w_grad_1, mask=feature_mask)
    tl.store(grad_weight_ptr + block_offset + weight_offset + 2, w_grad_2, mask=feature_mask)
    tl.store(grad_weight_ptr + block_offset + weight_offset + 3, w_grad_3, mask=feature_mask)
    tl.store(grad_weight_ptr + block_offset + weight_offset + 4, w_grad_4, mask=feature_mask)
    tl.store(grad_weight_ptr + block_offset + weight_offset + 5, w_grad_5, mask=feature_mask)
    tl.store(grad_weight_ptr + block_offset + weight_offset + 6, w_grad_6, mask=feature_mask)
    tl.store(grad_weight_ptr + block_offset + weight_offset + 7, w_grad_7, mask=feature_mask)
    tl.store(grad_weight_ptr + block_offset + weight_offset + 8, w_grad_8, mask=feature_mask)
    tl.store(grad_weight_ptr + block_offset + weight_offset + 9, w_grad_9, mask=feature_mask)
    tl.store(grad_weight_ptr + block_offset + weight_offset + 10, w_grad_10, mask=feature_mask)
    tl.store(grad_weight_ptr + block_offset + weight_offset + 11, w_grad_11, mask=feature_mask)
    tl.store(grad_weight_ptr + block_offset + weight_offset + 12, w_grad_12, mask=feature_mask)
    tl.store(grad_weight_ptr + block_offset + weight_offset + 13, w_grad_13, mask=feature_mask)
    tl.store(grad_weight_ptr + block_offset + weight_offset + 14, w_grad_14, mask=feature_mask)
    tl.store(grad_weight_ptr + block_offset + weight_offset + 15, w_grad_15, mask=feature_mask)
    tl.store(grad_weight_ptr + block_offset + weight_offset + 16, w_grad_16, mask=feature_mask)
    tl.store(grad_weight_ptr + block_offset + weight_offset + 17, w_grad_17, mask=feature_mask)
    tl.store(grad_weight_ptr + block_offset + weight_offset + 18, w_grad_18, mask=feature_mask)
    tl.store(grad_weight_ptr + block_offset + weight_offset + 19, w_grad_19, mask=feature_mask)


def gelu_geometric_product_norm_bwd(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    o: torch.Tensor,
    partial_norm: torch.Tensor,
    grad_output: torch.Tensor,
    normalize: bool,
) -> torch.Tensor:
    """Backward pass for the fused operation."""
    _, B, N = x.shape

    BATCH_BLOCK = min(DEFAULT_BATCH_BLOCK, B)
    FEATURE_BLOCK = min(DEFAULT_FEATURE_BLOCK, N)

    num_blocks_batch = triton.cdiv(B, BATCH_BLOCK)
    num_blocks_features = triton.cdiv(N, FEATURE_BLOCK)

    grad_x = torch.zeros_like(x)
    grad_y = torch.zeros_like(y)
    dot = (torch.zeros((NUM_GRADES, B), device=x.device, dtype=x.dtype) if normalize else torch.empty(0))
    grad_weight = torch.zeros((num_blocks_batch, N, NUM_PRODUCT_WEIGHTS), device=x.device, dtype=weight.dtype)

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
            EPS,
            num_warps=DEFAULT_NUM_WARPS,
            num_stages=DEFAULT_NUM_STAGES,
        )

    gelu_wgp_norm_kernel_bwd[grid](
        x,
        y,
        o,
        weight,
        dot,
        partial_norm,
        grad_output,
        grad_x,
        grad_y,
        grad_weight,
        normalize,
        B,
        N,
        BATCH_BLOCK,
        FEATURE_BLOCK,
        MV_DIM,
        NUM_GRADES,
        NUM_PRODUCT_WEIGHTS,
        EPS,
        num_warps=DEFAULT_NUM_WARPS,
        num_stages=DEFAULT_NUM_STAGES,
    )

    grad_weight = torch.sum(grad_weight, dim=0)

    return grad_x, grad_y, grad_weight


class WeightedGeluGeometricProductNorm3D(torch.autograd.Function):

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, y, weight, normalize):
        assert x.is_contiguous() and y.is_contiguous() and weight.is_contiguous()

        ctx.dtype = x.dtype
        ctx.normalize = normalize

        o, partial_norm = gelu_geometric_product_norm_fwd(
            x,
            y,
            weight,
            normalize,
        )

        ctx.save_for_backward(x, y, weight, o, partial_norm)

        return o.to(x.dtype)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()

        x, y, weight, o, partial_norm = ctx.saved_tensors

        grad_x, grad_y, grad_weight = gelu_geometric_product_norm_bwd(
            x,
            y,
            weight,
            o,
            partial_norm,
            grad_output,
            ctx.normalize,
        )

        return grad_x, grad_y, grad_weight, None, None, None, None


def fused_gelu_sgp_norm_3d(x, y, weight, normalize=True):
    """
    Fused operation that applies GELU non-linearity to two multivector inputs,
    then computes their weighted geometric product, and applies RMSNorm.
    
    Clifford algebra is assumed to be Cl(3,0).

    Args:
        x (torch.Tensor): Input tensor of shape (MV_DIM, B, N).
        y (torch.Tensor): Input tensor of shape (MV_DIM, B, N).
        weight (torch.Tensor): Weight tensor of shape (N, NUM_PRODUCT_WEIGHTS), one weight per geometric product component.
        normalize (bool): Whether to apply RMSNorm after the geometric product.

    Returns:
        torch.Tensor: Output tensor of shape (MV_DIM, B, N) after applying the fused operation.
    """
    return WeightedGeluGeometricProductNorm3D.apply(x, y, weight, normalize)