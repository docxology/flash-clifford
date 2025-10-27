# Core Concepts

This section covers the mathematical foundations underlying Flash Clifford, including Clifford algebra, multivectors, and the geometric product.

## Clifford Algebra

Clifford algebra Cl(p,q,r) is a mathematical structure that extends vector algebra by introducing a product operation that generalizes both the inner (dot) product and outer (wedge) product of vectors. It provides a unified framework for geometric computations in n-dimensional spaces.

### Multivectors

The fundamental objects in Clifford algebra are **multivectors** - linear combinations of basis elements:

```
A = Σᵢ₌₀ᵏ aᵢ eᵢ
```

where:
- `aᵢ` are scalar coefficients
- `eᵢ` are basis elements (e.g., scalars, vectors, bivectors)
- `k` is the dimension of the algebra

For Euclidean space Rⁿ, the Clifford algebra Cl(n,0) has 2ⁿ basis elements.

### Grades and Components

Multivectors are decomposed by **grade** - the dimensionality of the subspace spanned by the basis elements:

- **Grade 0 (Scalars)**: Real numbers, e.g., `a₀`
- **Grade 1 (Vectors)**: Direction and magnitude, e.g., `a₁e₁ + a₂e₂ + a₃e₃`
- **Grade 2 (Bivectors)**: Oriented planes, e.g., `a₁₂e₁₂ + a₁₃e₁₃ + a₂₃e₂₃`
- **Grade 3 (Trivectors)**: Oriented volumes, e.g., `a₁₂₃e₁₂₃`

Higher-grade elements encode increasingly complex geometric information.

## Geometric Product

The **geometric product** is the fundamental operation in Clifford algebra, combining inner and outer products:

```
AB = A · B + A ∧ B
```

where:
- `A · B` is the **inner product** (symmetric, scalar-valued)
- `A ∧ B` is the **outer product** (antisymmetric, higher-grade)

For vectors specifically: `ab = a · b + a ∧ b`

### Algebraic Properties

The geometric product satisfies several fundamental properties:

1. **Associativity**: `(AB)C = A(BC)`
2. **Distributivity**: `A(B + C) = AB + AC`
3. **Scalar preservation**: `λ(AB) = (λA)B = A(λB)`
4. **Involution**: `(AB)† = B†A†` (reverse operation)

### Vector Operations

For vectors `a, b ∈ ℝⁿ`, the geometric product encodes both dot and cross products:

```
ab = a · b + a ∧ b
```

where:
- `a · b = (1/2)(ab + ba)` is the standard dot product
- `a ∧ b = (1/2)(ab - ba)` is the wedge product (bivector)

### Magnitude and Inverse

The magnitude of a multivector is defined using the geometric product:

```
|A|² = AÃ
```

where `Ã` is the reverse (involution) of A. For invertible multivectors:

```
A⁻¹ = Ã / |A|²
```

### Equivariance

The geometric product preserves equivariance under orthogonal transformations. If `R` is an orthogonal transformation:

```
R(AB)R⁻¹ = (RA)(RB)
```

This property makes Clifford algebra particularly suitable for geometric deep learning, where preserving symmetries is crucial.

## Connection to Irreducible Representations

Clifford algebra components correspond directly to irreducible representations (irreps) of the orthogonal group O(n):

| Grade | Component Type | O(n) Irrep | Dimension |
|-------|----------------|------------|-----------|
| 0     | Scalar        | 0e         | 1         |
| 1     | Vector        | 1o         | n         |
| 2     | Bivector      | 1e         | n(n-1)/2  |
| 3     | Trivector     | 0o         | n(n-1)(n-2)/6 |

This correspondence enables efficient computation of equivariant neural networks.

## Implementation in Flash Clifford

### 2D Case (Cl(2,0))

For 2D Euclidean space, the multivector representation follows the standard basis:

```
[scalar, vector_x, vector_y, pseudoscalar] = [a₀, a₁, a₂, a₃]
```

The basis elements satisfy: `e₁² = e₂² = 1`, `e₁e₂ = -e₂e₁ = e₁₂` (pseudoscalar)

The geometric product between multivectors `A` and `B` implements the full algebraic structure:

```
AB = (a₀b₀ + a₁b₁ + a₂b₂ - a₃b₃,          # scalar component
      a₀b₁ + a₁b₀ - a₂b₃ + a₃b₂,          # vector_x component
      a₀b₂ + a₁b₃ + a₂b₀ - a₃b₁,          # vector_y component
      a₀b₃ + a₁b₂ - a₂b₁ + a₃b₀)          # pseudoscalar component
```

**Key observations:**
- Scalar component includes all grade-0 contributions
- Vector components mix grade-0 and grade-1 terms
- Pseudoscalar component captures the rotation effect (e₁e₂ = -e₂e₁)

### 3D Case (Cl(3,0))

For 3D Euclidean space, the multivector representation is:

```
[scalar, vector_x, vector_y, vector_z, bivector_xy, bivector_xz, bivector_yz, pseudoscalar]
```

The basis satisfies: `eᵢ² = 1`, `eᵢeⱼ = -eⱼeᵢ = eᵢⱼ` for i ≠ j

The geometric product implements the complete Cl(3,0) algebra with all 20 non-zero product weights:

**Scalar component:**
```
o₀ = w₀x₀y₀ + w₄(x₁y₁ + x₂y₂ + x₃y₃) - w₁₀(x₄y₄ + x₅y₅ + x₆y₆) - w₁₆x₇y₇
```

**Vector components:** Mix scalar, vector, bivector, and pseudoscalar terms
**Bivector components:** Capture rotations in coordinate planes
**Pseudoscalar component:** Full 3D rotation effects

### Algebraic Structure

The implementation hardcodes the complete Cayley table:

| Component | Grade | Transformation | Geometric Role |
|-----------|-------|----------------|----------------|
| 0 | 0 | Trivial | Scalar values |
| 1,2,3 | 1 | Vector | Translation equivariance |
| 4,5,6 | 2 | Bivector | Rotation equivariance |
| 7 | 3 | Pseudoscalar | Full rotation equivariance |

## Performance Considerations

The baseline approach represents the geometric product as a dense tensor contraction:

```
bni, mnijk, bnk → bmj
```

where `mnijk` is a sparse Cayley table encoding the product rules. While mathematically general, this approach is inefficient due to:

1. **Sparsity**: 85% zeros in 2D, 95% zeros in 3D
2. **Memory overhead**: Large tensor storage for sparse operations
3. **Computational waste**: Unnecessary operations on zero elements

Flash Clifford addresses these issues through hardcoded implementations that eliminate wasteful operations while preserving mathematical correctness.
