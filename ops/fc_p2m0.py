import torch
import triton
import triton.language as tl

MV_DIM = 4
NUM_GRADES = 3
NUM_PRODUCT_WEIGHTS = 10
WEIGHT_EXPANSION = [0, 3, 7, 1, 4, 5, 8, 1, 5, 4, 8, 2, 6, 9]
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
def gelu_pairwise_kernel_fwd(
    x_ptr,
    y_ptr,
    pairwise_ptr,
    batch_size: tl.constexpr,
    n_features: tl.constexpr,
    BATCH_BLOCK: tl.constexpr,
    FEATURE_BLOCK: tl.constexpr,
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

    stride_component = batch_size * n_features
    base_offset = batch_ids[:, None] * n_features + feature_ids[None, :]
    pairwise_offset = batch_ids[:, None] * n_features + feature_ids[None, :]

    x0 = tl.load(x_ptr + 0 * stride_component + base_offset, mask=batch_feature_mask)
    x1 = tl.load(x_ptr + 1 * stride_component + base_offset, mask=batch_feature_mask)
    x2 = tl.load(x_ptr + 2 * stride_component + base_offset, mask=batch_feature_mask)
    x3 = tl.load(x_ptr + 3 * stride_component + base_offset, mask=batch_feature_mask)

    y0 = tl.load(y_ptr + 0 * stride_component + base_offset, mask=batch_feature_mask)
    y1 = tl.load(y_ptr + 1 * stride_component + base_offset, mask=batch_feature_mask)
    y2 = tl.load(y_ptr + 2 * stride_component + base_offset, mask=batch_feature_mask)
    y3 = tl.load(y_ptr + 3 * stride_component + base_offset, mask=batch_feature_mask)

    gate_x = compute_gelu_gate(x0)
    gate_y = compute_gelu_gate(y0)

    x0 = x0 * gate_x
    x1 = x1 * gate_x
    x2 = x2 * gate_x
    x3 = x3 * gate_x

    y0 = y0 * gate_y
    y1 = y1 * gate_y
    y2 = y2 * gate_y
    y3 = y3 * gate_y

    p0 = x0 * y0
    p1 = x1*y1 + x2*y2
    p2 = -x3 * y3
    p3 = x0 * y1
    p4 = x1 * y0
    p5 = x2 * y3
    p6 = x3 * y2
    p7 = x0 * y2
    p8 = x1 * y3
    p9 = x2 * y0
    p10 = -x3 * y1
    p11 = x0 * y3
    p12 = x1*y2 - x2*y1
    p13 = x3 * y0

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
):
    """Gather linearly transformed pairwise products and compute the geometric product."""
    batch_block_id = tl.program_id(axis=0)
    thread_block_id = tl.program_id(axis=1)

    batch_ids = batch_block_id*BATCH_BLOCK + tl.arange(0, BATCH_BLOCK)
    feature_ids = thread_block_id*FEATURE_BLOCK + tl.arange(0, FEATURE_BLOCK)

    batch_mask = batch_ids < batch_size
    feature_mask = feature_ids < n_features
    batch_feature_mask = batch_mask[:, None] & feature_mask[None, :]

    stride_component = batch_size * n_features
    base_offset = batch_ids[:, None] * n_features + feature_ids[None, :]
    transformed_offset = batch_ids[:, None] * n_features + feature_ids[None, :]

    t0 = tl.load(transformed_ptr + 0*batch_size*n_features + transformed_offset, mask=batch_feature_mask)
    t1 = tl.load(transformed_ptr + 1*batch_size*n_features + transformed_offset, mask=batch_feature_mask)
    t2 = tl.load(transformed_ptr + 2*batch_size*n_features + transformed_offset, mask=batch_feature_mask)
    t3 = tl.load(transformed_ptr + 3*batch_size*n_features + transformed_offset, mask=batch_feature_mask)
    t4 = tl.load(transformed_ptr + 4*batch_size*n_features + transformed_offset, mask=batch_feature_mask)
    t5 = tl.load(transformed_ptr + 5*batch_size*n_features + transformed_offset, mask=batch_feature_mask)
    t6 = tl.load(transformed_ptr + 6*batch_size*n_features + transformed_offset, mask=batch_feature_mask)
    t7 = tl.load(transformed_ptr + 7*batch_size*n_features + transformed_offset, mask=batch_feature_mask)
    t8 = tl.load(transformed_ptr + 8*batch_size*n_features + transformed_offset, mask=batch_feature_mask)
    t9 = tl.load(transformed_ptr + 9*batch_size*n_features + transformed_offset, mask=batch_feature_mask)
    t10 = tl.load(transformed_ptr + 10*batch_size*n_features + transformed_offset, mask=batch_feature_mask)
    t11 = tl.load(transformed_ptr + 11*batch_size*n_features + transformed_offset, mask=batch_feature_mask)
    t12 = tl.load(transformed_ptr + 12*batch_size*n_features + transformed_offset, mask=batch_feature_mask)
    t13 = tl.load(transformed_ptr + 13*batch_size*n_features + transformed_offset, mask=batch_feature_mask)

    o0 = t0 + t1 + t2
    o1 = t3 + t4 - t5 + t6
    o2 = t7 + t8 + t9 + t10
    o3 = t11 + t12 + t13

    if NORMALIZE:
        pn_scalar = tl.sum(o0 * o0, axis=1) / n_features
        pn_vector = tl.sum(o1*o1 + o2*o2, axis=1) / n_features
        pn_pseudo = tl.sum(o3 * o3, axis=1) / n_features

        tl.atomic_add(pnorm_ptr + 0*batch_size + batch_ids, pn_scalar, mask=batch_mask)
        tl.atomic_add(pnorm_ptr + 1*batch_size + batch_ids, pn_vector, mask=batch_mask)
        tl.atomic_add(pnorm_ptr + 2*batch_size + batch_ids, pn_vector, mask=batch_mask)
        tl.atomic_add(pnorm_ptr + 3*batch_size + batch_ids, pn_pseudo, mask=batch_mask)

    tl.store(output_ptr + 0 * stride_component + base_offset, o0, mask=batch_feature_mask)
    tl.store(output_ptr + 1 * stride_component + base_offset, o1, mask=batch_feature_mask)
    tl.store(output_ptr + 2 * stride_component + base_offset, o2, mask=batch_feature_mask)
    tl.store(output_ptr + 3 * stride_component + base_offset, o3, mask=batch_feature_mask)


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


def gelu_fc_geometric_product_norm_fwd(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    expansion_indices: torch.Tensor,
    normalize: bool,
) -> torch.Tensor:
    """Fused operation: GELU non-linearity, fully connected geometric product, and grade-wise RMSNorm."""
    assert x.shape == y.shape
    assert x.shape[0] == MV_DIM
    assert x.shape[2] == weight.shape[1] == weight.shape[2]
    assert weight.shape[0] == NUM_PRODUCT_WEIGHTS

    _, B, N = x.shape

    BATCH_BLOCK = min(DEFAULT_BATCH_BLOCK, B)
    FEATURE_BLOCK = min(DEFAULT_FEATURE_BLOCK, N)

    num_blocks_batch = triton.cdiv(B, BATCH_BLOCK)
    num_blocks_features = triton.cdiv(N, FEATURE_BLOCK)

    pairwise = torch.empty((len(WEIGHT_EXPANSION), B, N), device=x.device, dtype=x.dtype)
    partial_norm = (torch.zeros((MV_DIM, B), device=x.device, dtype=x.dtype) if normalize else torch.zeros((1,), device=x.device, dtype=x.dtype))
    output = torch.empty_like(x)

    grid = (num_blocks_batch, num_blocks_features)

    gelu_pairwise_kernel_fwd[grid](
        x,
        y,
        pairwise,
        B,
        N,
        BATCH_BLOCK,
        FEATURE_BLOCK,
        num_warps=DEFAULT_NUM_WARPS,
        num_stages=DEFAULT_NUM_STAGES,
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

    pn_scalar = tl.load(pnorm_ptr + 0*batch_size + batch_ids, mask=batch_mask)[:, None]
    pn_vector = tl.load(pnorm_ptr + 1*batch_size + batch_ids, mask=batch_mask)[:, None]
    pn_pseudo = tl.load(pnorm_ptr + 3*batch_size + batch_ids, mask=batch_mask)[:, None]

    o0 = tl.load(output_ptr + 0 * stride_component + offset, mask=batch_feature_mask)
    o1 = tl.load(output_ptr + 1 * stride_component + offset, mask=batch_feature_mask)
    o2 = tl.load(output_ptr + 2 * stride_component + offset, mask=batch_feature_mask)
    o3 = tl.load(output_ptr + 3 * stride_component + offset, mask=batch_feature_mask)

    rms_scalar = tl.sqrt(pn_scalar + EPS)
    rms_vector = tl.sqrt(pn_vector + EPS)
    rms_pseudo = tl.sqrt(pn_pseudo + EPS)

    dot_scalar = tl.sum(rms_scalar * go0 * o0, axis=1)
    dot_vector = tl.sum(rms_vector * (go1*o1 + go2*o2), axis=1)
    dot_pseudo = tl.sum(rms_pseudo * go3 * o3, axis=1)

    tl.atomic_add(dot_ptr + 0*batch_size + batch_ids, dot_scalar, mask=batch_mask)
    tl.atomic_add(dot_ptr + 1*batch_size + batch_ids, dot_vector, mask=batch_mask)
    tl.atomic_add(dot_ptr + 2*batch_size + batch_ids, dot_pseudo, mask=batch_mask)


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

    stride_component = batch_size * n_features
    base_offset = batch_ids[:, None] * n_features + feature_ids[None, :]
    transformed_offset = batch_ids[:, None] * n_features + feature_ids[None, :]

    go0 = tl.load(grad_output_ptr + 0 * stride_component + base_offset, mask=batch_feature_mask)
    go1 = tl.load(grad_output_ptr + 1 * stride_component + base_offset, mask=batch_feature_mask)
    go2 = tl.load(grad_output_ptr + 2 * stride_component + base_offset, mask=batch_feature_mask)
    go3 = tl.load(grad_output_ptr + 3 * stride_component + base_offset, mask=batch_feature_mask)

    if NORMALIZE:
        o0 = tl.load(output_ptr + 0 * stride_component + base_offset, mask=batch_feature_mask)
        o1 = tl.load(output_ptr + 1 * stride_component + base_offset, mask=batch_feature_mask)
        o2 = tl.load(output_ptr + 2 * stride_component + base_offset, mask=batch_feature_mask)
        o3 = tl.load(output_ptr + 3 * stride_component + base_offset, mask=batch_feature_mask)

        pn_scalar = tl.load(pnorm_ptr + 0*batch_size + batch_ids, mask=batch_mask)[:, None]
        pn_vector = tl.load(pnorm_ptr + 1*batch_size + batch_ids, mask=batch_mask)[:, None]
        pn_pseudo = tl.load(pnorm_ptr + 3*batch_size + batch_ids, mask=batch_mask)[:, None]

        dot_scalar = tl.load(dot_ptr + 0*batch_size + batch_ids, mask=batch_mask)[:, None]
        dot_vector = tl.load(dot_ptr + 1*batch_size + batch_ids, mask=batch_mask)[:, None]
        dot_pseudo = tl.load(dot_ptr + 2*batch_size + batch_ids, mask=batch_mask)[:, None]

        rms_scalar = tl.sqrt(pn_scalar + EPS)
        rms_vector = tl.sqrt(pn_vector + EPS)
        rms_pseudo = tl.sqrt(pn_pseudo + EPS)

        go0 = go0/rms_scalar - o0 * dot_scalar / (n_features*rms_scalar*rms_scalar)
        go1 = go1/rms_vector - o1 * dot_vector / (n_features*rms_vector*rms_vector)
        go2 = go2/rms_vector - o2 * dot_vector / (n_features*rms_vector*rms_vector)
        go3 = go3/rms_pseudo - o3 * dot_pseudo / (n_features*rms_pseudo*rms_pseudo)

    tl.store(grad_transformed_ptr + 0*batch_size*n_features + transformed_offset, go0, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 1*batch_size*n_features + transformed_offset, go0, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 2*batch_size*n_features + transformed_offset, go0, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 3*batch_size*n_features + transformed_offset, go1, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 4*batch_size*n_features + transformed_offset, go1, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 5*batch_size*n_features + transformed_offset, -go1, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 6*batch_size*n_features + transformed_offset, go1, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 7*batch_size*n_features + transformed_offset, go2, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 8*batch_size*n_features + transformed_offset, go2, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 9*batch_size*n_features + transformed_offset, go2, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 10*batch_size*n_features + transformed_offset, go2, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 11*batch_size*n_features + transformed_offset, go3, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 12*batch_size*n_features + transformed_offset, go3, mask=batch_feature_mask)
    tl.store(grad_transformed_ptr + 13*batch_size*n_features + transformed_offset, go3, mask=batch_feature_mask)


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

    stride_component = batch_size * n_features
    base_offset = batch_ids[:, None] * n_features + feature_ids[None, :]
    pairwise_offset = batch_ids[:, None] * n_features + feature_ids[None, :]
    
    gp0 = tl.load(grad_pairwise_ptr + 0*batch_size*n_features + pairwise_offset, mask=batch_feature_mask)
    gp1 = tl.load(grad_pairwise_ptr + 1*batch_size*n_features + pairwise_offset, mask=batch_feature_mask)
    gp2 = tl.load(grad_pairwise_ptr + 2*batch_size*n_features + pairwise_offset, mask=batch_feature_mask)
    gp3 = tl.load(grad_pairwise_ptr + 3*batch_size*n_features + pairwise_offset, mask=batch_feature_mask)
    gp4 = tl.load(grad_pairwise_ptr + 4*batch_size*n_features + pairwise_offset, mask=batch_feature_mask)
    gp5 = tl.load(grad_pairwise_ptr + 5*batch_size*n_features + pairwise_offset, mask=batch_feature_mask)
    gp6 = tl.load(grad_pairwise_ptr + 6*batch_size*n_features + pairwise_offset, mask=batch_feature_mask)
    gp7 = tl.load(grad_pairwise_ptr + 7*batch_size*n_features + pairwise_offset, mask=batch_feature_mask)
    gp8 = tl.load(grad_pairwise_ptr + 8*batch_size*n_features + pairwise_offset, mask=batch_feature_mask)
    gp9 = tl.load(grad_pairwise_ptr + 9*batch_size*n_features + pairwise_offset, mask=batch_feature_mask)
    gp10 = tl.load(grad_pairwise_ptr + 10*batch_size*n_features + pairwise_offset, mask=batch_feature_mask)
    gp11 = tl.load(grad_pairwise_ptr + 11*batch_size*n_features + pairwise_offset, mask=batch_feature_mask)
    gp12 = tl.load(grad_pairwise_ptr + 12*batch_size*n_features + pairwise_offset, mask=batch_feature_mask)
    gp13 = tl.load(grad_pairwise_ptr + 13*batch_size*n_features + pairwise_offset, mask=batch_feature_mask)

    x0_raw = tl.load(x_ptr + 0 * stride_component + base_offset, mask=batch_feature_mask)
    x1_raw = tl.load(x_ptr + 1 * stride_component + base_offset, mask=batch_feature_mask)
    x2_raw = tl.load(x_ptr + 2 * stride_component + base_offset, mask=batch_feature_mask)
    x3_raw = tl.load(x_ptr + 3 * stride_component + base_offset, mask=batch_feature_mask)

    y0_raw = tl.load(y_ptr + 0 * stride_component + base_offset, mask=batch_feature_mask)
    y1_raw = tl.load(y_ptr + 1 * stride_component + base_offset, mask=batch_feature_mask)
    y2_raw = tl.load(y_ptr + 2 * stride_component + base_offset, mask=batch_feature_mask)
    y3_raw = tl.load(y_ptr + 3 * stride_component + base_offset, mask=batch_feature_mask)

    # collect gradients from pairwise products
    gate_x = compute_gelu_gate(x0_raw)
    gate_y = compute_gelu_gate(y0_raw)

    x0 = x0_raw * gate_x
    x1 = x1_raw * gate_x
    x2 = x2_raw * gate_x
    x3 = x3_raw * gate_x

    y0 = y0_raw * gate_y
    y1 = y1_raw * gate_y
    y2 = y2_raw * gate_y
    y3 = y3_raw * gate_y

    x_grad_0 = gp0*y0 + gp3*y1 + gp7*y2 + gp11*y3
    x_grad_1 = gp1*y1 + gp4*y0 + gp8*y3 + gp12*y2
    x_grad_2 = gp1*y2 + gp5*y3 + gp9*y0 - gp12*y1
    x_grad_3 = -gp2 * y3 + gp6*y2 - gp10*y1 + gp13*y0

    y_grad_0 = gp0*x0 + gp4*x1 + gp9*x2 + gp13*x3
    y_grad_1 = gp1*x1 + gp3*x0 - gp10*x3 - gp12*x2
    y_grad_2 = gp1*x2 + gp6*x3 + gp7*x0 + gp12*x1
    y_grad_3 = -gp2 * x3 + gp5*x2 + gp8*x1 + gp11*x0
    
    # GELU gate gradients
    dgate_x = compute_gelu_gate_grad(x0_raw)
    dgate_y = compute_gelu_gate_grad(y0_raw)

    x_grad_0 = (gate_x + x0_raw*dgate_x) * x_grad_0 + dgate_x * (x1_raw*x_grad_1 + x2_raw*x_grad_2 + x3_raw*x_grad_3)
    x_grad_1 = gate_x * x_grad_1
    x_grad_2 = gate_x * x_grad_2
    x_grad_3 = gate_x * x_grad_3

    y_grad_0 = (gate_y + y0_raw*dgate_y) * y_grad_0 + dgate_y * (y1_raw*y_grad_1 + y2_raw*y_grad_2 + y3_raw*y_grad_3)
    y_grad_1 = gate_y * y_grad_1
    y_grad_2 = gate_y * y_grad_2
    y_grad_3 = gate_y * y_grad_3

    tl.store(grad_x_ptr + 0 * stride_component + base_offset, x_grad_0, mask=batch_feature_mask)
    tl.store(grad_x_ptr + 1 * stride_component + base_offset, x_grad_1, mask=batch_feature_mask)
    tl.store(grad_x_ptr + 2 * stride_component + base_offset, x_grad_2, mask=batch_feature_mask)
    tl.store(grad_x_ptr + 3 * stride_component + base_offset, x_grad_3, mask=batch_feature_mask)

    tl.store(grad_y_ptr + 0 * stride_component + base_offset, y_grad_0, mask=batch_feature_mask)
    tl.store(grad_y_ptr + 1 * stride_component + base_offset, y_grad_1, mask=batch_feature_mask)
    tl.store(grad_y_ptr + 2 * stride_component + base_offset, y_grad_2, mask=batch_feature_mask)
    tl.store(grad_y_ptr + 3 * stride_component + base_offset, y_grad_3, mask=batch_feature_mask)


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
            EPS,
            num_warps=DEFAULT_NUM_WARPS,
            num_stages=DEFAULT_NUM_STAGES,
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
        EPS,
        num_warps=DEFAULT_NUM_WARPS,
        num_stages=DEFAULT_NUM_STAGES,
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
        num_warps=DEFAULT_NUM_WARPS,
        num_stages=DEFAULT_NUM_STAGES,
    )

    return grad_x, grad_y, grad_weight


class FullyConnectedGeluGeometricProductNorm2D(torch.autograd.Function):

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, y, weight, normalize):
        assert x.is_contiguous() and y.is_contiguous() and weight.is_contiguous()

        ctx.dtype = x.dtype
        ctx.normalize = normalize

        expansion_indices = torch.tensor(WEIGHT_EXPANSION, device=x.device)

        o, pairwise, partial_norm = gelu_fc_geometric_product_norm_fwd(
            x,
            y,
            weight,
            expansion_indices,
            normalize,
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
        )

        return grad_x, grad_y, grad_weight, None, None, None, None


def fused_gelu_fc_sgp_norm_2d(x, y, weight, normalize=True):
    """
    Fused operation that applies GELU non-linearity to two multivector inputs,
    then computes their fully connected geometric product, and applies RMSNorm.

    Args:
        x (torch.Tensor): Input tensor of shape (MV_DIM, B, N).
        y (torch.Tensor): Input tensor of shape (MV_DIM, B, N).
        weight (torch.Tensor): Weight tensor of shape (NUM_PRODUCT_WEIGHTS, N, N), one weight per geometric product component.
        normalize (bool): Whether to apply RMSNorm after the geometric product.

    Returns:
        torch.Tensor: Output tensor of shape (MV_DIM, B, N) after applying the fused operation.
    """
    return FullyConnectedGeluGeometricProductNorm2D.apply(x, y, weight, normalize)
