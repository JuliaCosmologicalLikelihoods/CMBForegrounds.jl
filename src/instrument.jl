"""
    instrument.jl

Instrumental operations and systematic effects:
1. Calibration: map gains, independent spectrum gains, forward vs inverse conventions.
2. Additive templates: fixed and variable amplitude template perturbations.
3. Polarization leakage: map-response algebra for T -> E leakage (TE, ET, and EE).
4. Beam modes: linear and quadratic beam eigenmode perturbations.
5. SSL & aberration: Super-Sample Lensing and relativistic aberration response composition.

The operations are non-mutating. Differentiability is tested for the parameterized
paths used by the public component and instrument interfaces.
"""

# ------------------------------------------------------------------ #
# 1. Calibration operations                                           #
# ------------------------------------------------------------------ #

"""
    calibration_factor(c_i::Real, c_j::Real; convention::Symbol=:forward)

Compute the effective calibration factor for pair `(i, j)`.
- `:forward` (default): returns `c_i * c_j` (model spectrum scaled to match data: ``C_\\ell^\\mathrm{obs} = c_i c_j C_\\ell^\\mathrm{true}``)
- `:inverse`: returns `1 / (c_i * c_j)` (data calibrated to sky: ``C_\\ell^\\mathrm{cal} = C_\\ell^\\mathrm{obs} / (c_i c_j)``)
"""
@inline function calibration_factor(c_i::Real, c_j::Real; convention::Symbol=:forward)
    if convention === :forward
        return c_i * c_j
    elseif convention === :inverse
        return inv(c_i * c_j)
    else
        throw(ArgumentError("calibration convention must be :forward or :inverse, got :$convention"))
    end
end

"""
    apply_calibration(cl::AbstractVector, c::Real; convention::Symbol=:forward)

Apply scalar calibration factor `c` to power spectrum `cl`.
"""
@inline function apply_calibration(cl::AbstractVector, c::Real; convention::Symbol=:forward)
    if convention === :forward
        return cl .* c
    elseif convention === :inverse
        return cl ./ c
    else
        throw(ArgumentError("calibration convention must be :forward or :inverse, got :$convention"))
    end
end

"""
    apply_calibration(cl::AbstractVector, c_i::Real, c_j::Real; convention::Symbol=:forward)

Apply map calibration factors `c_i` and `c_j` to cross-spectrum `cl`.
"""
@inline function apply_calibration(cl::AbstractVector, c_i::Real, c_j::Real; convention::Symbol=:forward)
    f = calibration_factor(c_i, c_j; convention=convention)
    return cl .* f
end

"""
    apply_calibration(D::AbstractArray{<:Any, 3}, gains::AbstractVector; convention::Symbol=:forward)

Apply channel/map calibration factors `gains` (length `n_freq`) to 3D power spectrum array `D` of shape `(n_freq, n_freq, n_ell)`.
"""
function apply_calibration(D::AbstractArray{<:Any, 3}, gains::AbstractVector; convention::Symbol=:forward)
    n_freq = size(D, 1)
    @assert size(D, 2) == n_freq "apply_calibration: D must have square frequency dimensions"
    @assert length(gains) == n_freq "apply_calibration: gains length must match D frequency dimensions"

    G = gains .* transpose(gains)
    if convention === :inverse
        G = inv.(G)
    elseif convention !== :forward
        throw(ArgumentError("calibration convention must be :forward or :inverse, got :$convention"))
    end

    n_ell = size(D, 3)
    G_3d = reshape(G, n_freq, n_freq, 1)
    return D .* G_3d
end

"""
    apply_calibration(D::AbstractArray{<:Any, 3}, gains::AbstractMatrix; convention::Symbol=:forward)

Apply independent pair/spectrum calibration factors `gains` of shape `(n_freq, n_freq)` to 3D spectrum array `D`.
"""
function apply_calibration(D::AbstractArray{<:Any, 3}, gains::AbstractMatrix; convention::Symbol=:forward)
    n_freq = size(D, 1)
    @assert size(D, 2) == n_freq "apply_calibration: D must have square frequency dimensions"
    @assert size(gains) == (n_freq, n_freq) "apply_calibration: gains matrix shape must match D frequency dimensions"

    G = convention === :forward ? gains : (convention === :inverse ? inv.(gains) : throw(ArgumentError("calibration convention must be :forward or :inverse, got :$convention")))
    n_ell = size(D, 3)
    G_3d = reshape(G, n_freq, n_freq, 1)
    return D .* G_3d
end

# ------------------------------------------------------------------ #
# 2. Additive templates                                               #
# ------------------------------------------------------------------ #

"""
    additive_template(template::AbstractVector, amp::Real=1.0)

Scale an additive multipole template `template` by numerical amplitude `amp` (fixed or variable).
"""
@inline additive_template(template::AbstractVector, amp::Real=1.0) = amp .* template

"""
    additive_template(template::AbstractVector, amp_matrix::AbstractMatrix)

Construct a 3D additive template tensor `(n_freq, n_freq, n_ell)` from a 1D template and a 2D amplitude matrix `amp_matrix`.
"""
function additive_template(template::AbstractVector, amp_matrix::AbstractMatrix)
    n_freq = size(amp_matrix, 1)
    @assert size(amp_matrix, 2) == n_freq "amp_matrix must be square"
    n_ell = length(template)
    return reshape(amp_matrix, n_freq, n_freq, 1) .* reshape(template, 1, 1, n_ell)
end

"""
    add_template(cl::AbstractVector, template::AbstractVector, amp::Real=1.0)

Add a scaled multipole template `amp * template` to spectrum `cl`.
"""
@inline function add_template(cl::AbstractVector, template::AbstractVector, amp::Real=1.0)
    length(cl) == length(template) ||
        throw(DimensionMismatch("cl and template must have the same length"))
    return cl .+ (amp .* template)
end

"""
    add_template(D::AbstractArray{<:Any, 3}, template::AbstractVector, amp_matrix::AbstractMatrix)

Add an amplitude-weighted template to a 3D spectrum tensor `D`.
"""
function add_template(D::AbstractArray{<:Any, 3}, template::AbstractVector, amp_matrix::AbstractMatrix)
    size(D, 1) == size(D, 2) ||
        throw(DimensionMismatch("D must have square frequency dimensions"))
    length(template) == size(D, 3) ||
        throw(DimensionMismatch("template length must match the multipole dimension of D"))
    size(amp_matrix) == (size(D, 1), size(D, 2)) ||
        throw(DimensionMismatch("amp_matrix must match the frequency dimensions of D"))
    n_freq = size(D, 1)
    n_ell = size(D, 3)
    return D .+ reshape(amp_matrix, n_freq, n_freq, 1) .*
                reshape(template, 1, 1, n_ell)
end

# ------------------------------------------------------------------ #
# 3. Polarization leakage map-response algebra                        #
# ------------------------------------------------------------------ #

"""
    te_leakage(C_TT::AbstractVector, gamma_j::Union{Real, AbstractVector})

Compute the additive T -> E polarization leakage contribution to TE:
```math
\\Delta C_\\ell^{TE}(i, j) = \\gamma_j(\\ell) \\cdot C_\\ell^{TT}(i, j)
```
where `gamma_j` is the leakage factor/curve for channel `j`.
"""
@inline function te_leakage(C_TT::AbstractVector, gamma_j::Union{Real, AbstractVector})
    gamma_j isa AbstractVector && length(gamma_j) != length(C_TT) &&
        throw(DimensionMismatch("gamma_j must have the same length as C_TT"))
    return gamma_j .* C_TT
end

"""
    et_leakage(C_TT::AbstractVector, gamma_i::Union{Real, AbstractVector})

Compute the additive T -> E polarization leakage contribution to ET:
```math
\\Delta C_\\ell^{ET}(i, j) = \\gamma_i(\\ell) \\cdot C_\\ell^{TT}(i, j)
```
where `gamma_i` is the leakage factor/curve for channel `i`.
"""
@inline function et_leakage(C_TT::AbstractVector, gamma_i::Union{Real, AbstractVector})
    gamma_i isa AbstractVector && length(gamma_i) != length(C_TT) &&
        throw(DimensionMismatch("gamma_i must have the same length as C_TT"))
    return gamma_i .* C_TT
end

"""
    ee_leakage(C_TT::AbstractVector, C_TE_ij::AbstractVector, C_TE_ji::AbstractVector,
               gamma_i::Union{Real, AbstractVector}, gamma_j::Union{Real, AbstractVector})

Compute the full map-response algebra leakage contribution to EE cross-spectrum:
```math
\\Delta C_\\ell^{EE}(i, j) = \\gamma_i(\\ell) C_\\ell^{TE}(i, j) + \\gamma_j(\\ell) C_\\ell^{TE}(j, i) + \\gamma_i(\\ell) \\gamma_j(\\ell) C_\\ell^{TT}(i, j)
```
Retains both linear cross-terms and the quadratic TT term.
"""
function ee_leakage(C_TT::AbstractVector, C_TE_ij::AbstractVector, C_TE_ji::AbstractVector,
                    gamma_i::Union{Real, AbstractVector}, gamma_j::Union{Real, AbstractVector})
    n_ell = length(C_TT)
    length(C_TE_ij) == n_ell && length(C_TE_ji) == n_ell ||
        throw(DimensionMismatch("TT, TE_ij, and TE_ji must have the same length"))
    gamma_i isa AbstractVector && length(gamma_i) != n_ell &&
        throw(DimensionMismatch("gamma_i must have the same length as the spectra"))
    gamma_j isa AbstractVector && length(gamma_j) != n_ell &&
        throw(DimensionMismatch("gamma_j must have the same length as the spectra"))
    return @. gamma_i * C_TE_ij + gamma_j * C_TE_ji + (gamma_i * gamma_j) * C_TT
end

"""
    ee_leakage(C_TT::AbstractVector, C_TE::AbstractVector, gamma::Union{Real, AbstractVector})

Compute auto-spectrum EE leakage (i = j):
```math
\\Delta C_\\ell^{EE}(i, i) = 2 \\gamma(\\ell) C_\\ell^{TE}(i, i) + \\gamma(\\ell)^2 C_\\ell^{TT}(i, i)
```
"""
@inline function ee_leakage(C_TT::AbstractVector, C_TE::AbstractVector, gamma::Union{Real, AbstractVector})
    length(C_TE) == length(C_TT) ||
        throw(DimensionMismatch("C_TT and C_TE must have the same length"))
    gamma isa AbstractVector && length(gamma) != length(C_TT) &&
        throw(DimensionMismatch("gamma must have the same length as the spectra"))
    return @. 2 * gamma * C_TE + (gamma * gamma) * C_TT
end

"""
    apply_te_leakage(D_TE::AbstractArray{<:Any,3}, D_TT::AbstractArray{<:Any,3}, gammas::AbstractVector)

Apply map-level leakage vectors `gammas` (length `n_freq`) across 3D spectrum tensors:
```math
D_\\ell^{TE,\\mathrm{obs}}[i, j] = D_\\ell^{TE}[i, j] + \\gamma_j \\cdot D_\\ell^{TT}[i, j]
```
"""
function apply_te_leakage(D_TE::AbstractArray{<:Any,3}, D_TT::AbstractArray{<:Any,3}, gammas::AbstractVector)
    n_freq = size(D_TE, 1)
    size(D_TE, 2) == n_freq ||
        throw(DimensionMismatch("D_TE must have square frequency dimensions"))
    size(D_TT) == size(D_TE) ||
        throw(DimensionMismatch("D_TT and D_TE must have identical dimensions"))
    length(gammas) == n_freq ||
        throw(DimensionMismatch("gammas length must match the frequency dimensions"))
    G_j = reshape(gammas, 1, n_freq, 1)
    return D_TE .+ (G_j .* D_TT)
end

"""
    apply_ee_leakage(D_EE::AbstractArray{<:Any,3}, D_TE::AbstractArray{<:Any,3}, D_TT::AbstractArray{<:Any,3}, gammas::AbstractVector)

Apply map-level leakage across 3D tensors:
```math
D_\\ell^{EE,\\mathrm{obs}}[i, j] = D_\\ell^{EE}[i, j] + \\gamma_i D_\\ell^{TE}[i, j] + \\gamma_j D_\\ell^{TE}[j, i] + \\gamma_i \\gamma_j D_\\ell^{TT}[i, j]
```
`D_TE` must be the underlying unleaked ordered TE tensor. Passing the output of
`apply_te_leakage` here double-counts part of the response.
"""
function apply_ee_leakage(D_EE::AbstractArray{<:Any,3}, D_TE::AbstractArray{<:Any,3}, D_TT::AbstractArray{<:Any,3}, gammas::AbstractVector)
    n_freq = size(D_EE, 1)
    size(D_EE, 2) == n_freq ||
        throw(DimensionMismatch("D_EE must have square frequency dimensions"))
    size(D_TE) == size(D_EE) && size(D_TT) == size(D_EE) ||
        throw(DimensionMismatch("D_EE, D_TE, and D_TT must have identical dimensions"))
    length(gammas) == n_freq ||
        throw(DimensionMismatch("gammas length must match the frequency dimensions"))
    G_i = reshape(gammas, n_freq, 1, 1)
    G_j = reshape(gammas, 1, n_freq, 1)
    D_ET = PermutedDimsArray(D_TE, (2, 1, 3))
    return @. D_EE + G_i * D_TE + G_j * D_ET + (G_i * G_j) * D_TT
end

# ------------------------------------------------------------------ #
# 4. Beam eigenmode perturbations                                    #
# ------------------------------------------------------------------ #

"""
    beam_eigenmode_response(cl::AbstractVector, modes::AbstractMatrix, coeffs::AbstractVector; linearized::Bool=false)

Compute the beam perturbation to power spectrum `cl` given beam eigenmodes `modes` (shape `(n_ell, n_modes)`)
and mode amplitudes `coeffs` (length `n_modes`).

The mode expansion is interpreted as a fractional map-beam perturbation. Modes
already expressed as power-spectrum or window-function errors use a different
amplitude convention and must not be passed directly.

- If `linearized` is `true`, return the first-order corrected spectrum:
  ```math
  C_\\ell^\\mathrm{perturbed} = C_\\ell \\left(1 + 2\\sum_k \\beta_k \\phi_k(\\ell)\\right)
  ```
- If `linearized` is `false`:
  ```math
  C_\\ell^\\mathrm{perturbed} = C_\\ell \\left(1 + \\sum_k \\beta_k \\phi_k(\\ell)\\right)^2
  ```
"""
function beam_eigenmode_response(cl::AbstractVector, modes::AbstractMatrix, coeffs::AbstractVector; linearized::Bool=false)
    n_ell = length(cl)
    @assert size(modes, 1) == n_ell "modes rows must match cl length"
    @assert size(modes, 2) == length(coeffs) "modes columns must match coeffs length"

    delta_b = modes * coeffs
    if linearized
        return @. cl * (1 + 2 * delta_b)
    else
        return @. cl * (1 + delta_b) * (1 + delta_b)
    end
end

"""
    beam_eigenmode_cross(cl::AbstractVector, modes_i::AbstractMatrix, coeffs_i::AbstractVector,
                         modes_j::AbstractMatrix, coeffs_j::AbstractVector; linearized::Bool=false)

Compute beam eigenmode perturbation for cross-spectrum between channel `i` and channel `j`:
- `linearized == true`: returns ``C_\\ell [1 + \\Delta B_i(\\ell) + \\Delta B_j(\\ell)]``
- `linearized == false`: returns ``C_\\ell [1 + \\Delta B_i(\\ell)] [1 + \\Delta B_j(\\ell)]``
"""
function beam_eigenmode_cross(cl::AbstractVector, modes_i::AbstractMatrix, coeffs_i::AbstractVector,
                              modes_j::AbstractMatrix, coeffs_j::AbstractVector; linearized::Bool=false)
    n_ell = length(cl)
    @assert size(modes_i, 1) == n_ell && size(modes_j, 1) == n_ell
    @assert size(modes_i, 2) == length(coeffs_i) && size(modes_j, 2) == length(coeffs_j)

    delta_bi = modes_i * coeffs_i
    delta_bj = modes_j * coeffs_j
    if linearized
        return @. cl * (1 + delta_bi + delta_bj)
    else
        return @. cl * (1 + delta_bi) * (1 + delta_bj)
    end
end

# ------------------------------------------------------------------ #
# 5. SSL and Aberration Compositions                                  #
# ------------------------------------------------------------------ #

"""
    apply_ssl(ells::AbstractVector, κ::Real, Dℓ::AbstractVector)

Apply Super-Sample Lensing (SSL) correction to power spectrum `Dℓ`:
```math
D_\\ell^\\mathrm{corrected} = D_\\ell + \\mathrm{ssl\\_response}(\\ell, \\kappa, D_\\ell)
```
"""
@inline function apply_ssl(ells::AbstractVector, κ::Real, Dℓ::AbstractVector)
    return Dℓ .+ ssl_response(ells, κ, Dℓ)
end

"""
    apply_aberration(ells::AbstractVector, ab_coeff::Real, Dℓ::AbstractVector)

Apply relativistic aberration correction to power spectrum `Dℓ`:
```math
D_\\ell^\\mathrm{corrected} = D_\\ell + \\mathrm{aberration\\_response}(\\ell, A_\\mathrm{aberr}, D_\\ell)
```
"""
@inline function apply_aberration(ells::AbstractVector, ab_coeff::Real, Dℓ::AbstractVector)
    return Dℓ .+ aberration_response(ells, ab_coeff, Dℓ)
end
