"""
    correlation.jl

Cross-correlation models and multiple-dispatch interface.

Provides mathematical representations for correlations between sky components
(e.g., thermal SZ × cosmic infrared background):
- `TemplateCorrelation`: signed cross-template prescription (Planck, HiLLiPoP, ACT DR6)
  ```math
  D_\\ell^{1 \\times 2}(\\nu_1, \\nu_2) = -\\xi \\sqrt{|A_1 A_2|} \\bigl[f_{1}(\\nu_1) f_{2}(\\nu_2) + f_{1}(\\nu_2) f_{2}(\\nu_1)\\bigr] \\cdot T_\\ell
  ```
  Retains the physical signed SED factors (e.g. tSZ passing through its null at 217 GHz).
- `GeometricMeanCorrelation`: geometric-mean cross prescription (SPT-3G 2018)
  ```math
  D_\\ell^{1 \\times 2}(\\nu_1, \\nu_2) = -\\xi \\left( \\sqrt{|D_\\ell^{1, 11} D_\\ell^{2, 22}|} + \\sqrt{|D_\\ell^{1, 22} D_\\ell^{2, 11}|} \\right)
  ```
  Constructed from auto-spectra; always non-positive for ξ ≥ 0 (valid where both frequencies
  are below the tSZ null).

Both evaluate through the unified `correlation_power` interface.
"""

"""
    AbstractCorrelationModel

Abstract supertype for all cross-component correlation representations.
"""
abstract type AbstractCorrelationModel end

"""
    TemplateCorrelation{S<:AbstractAngularModel} <: AbstractCorrelationModel

Signed template-based cross-correlation model between two sky components.

# Model
```math
D_\\ell^{1 \\times 2}(\\nu_1, \\nu_2) = -\\xi \\sqrt{|A_1 A_2|}
    \\bigl[f_{1}(\\nu_1) f_{2}(\\nu_2) + f_{1}(\\nu_2) f_{2}(\\nu_1)\\bigr] \\cdot T_\\ell
```

# Fields
- `shape::S`: Angular cross-template representation (`TemplateShape` or other `AbstractAngularModel`).

Angular parameters such as a power-law slope or template tilt are passed as
keywords to `correlation_power`. The ell-free convenience evaluator is available
only for `TemplateShape`, whose multipole grid is stored in the template itself.
"""
struct TemplateCorrelation{S<:AbstractAngularModel} <: AbstractCorrelationModel
    shape::S
end

function TemplateCorrelation(template::AbstractVector)
    return TemplateCorrelation(TemplateShape(template; ell_0=nothing, ell_min=0))
end

"""
    GeometricMeanCorrelation <: AbstractCorrelationModel

Geometric-mean cross-correlation model between two sky components.

# Model
```math
D_\\ell^{1 \\times 2}(\\nu_1, \\nu_2) = -\\xi
    \\left( \\sqrt{|D_\\ell^{1, 11} D_\\ell^{2, 22}|} + \\sqrt{|D_\\ell^{1, 22} D_\\ell^{2, 11}|} \\right)
```

# Notes & Domain
- Auto-spectra are inherently positive, so this model does not preserve the frequency
  sign change of tSZ across its null (ν ≈ 217.4 GHz). It is intended for frequency regimes
  below the null (e.g., SPT-3G baseline bands).
- The `abs` inside the square root provides a guard against transient negative values in
  MCMC/HMC samplers, but note that the derivative at A = 0 has a branch point.
"""
struct GeometricMeanCorrelation <: AbstractCorrelationModel end

# ------------------------------------------------------------------ #
# Common evaluation interface: correlation_power                       #
# ------------------------------------------------------------------ #

"""
    correlation_power(corr::AbstractCorrelationModel, args...; kwargs...)

Evaluate the cross-correlation power spectrum ``D_\\ell`` for the given correlation model.
"""
function correlation_power end

# TemplateCorrelation evaluator
@inline function correlation_power(corr::TemplateCorrelation, ells::AbstractVector,
                                   xi::Real, A1::Real, A2::Real,
                                   f1_1::Union{Real, AbstractVector},
                                   f1_2::Union{Real, AbstractVector},
                                   f2_1::Union{Real, AbstractVector},
                                   f2_2::Union{Real, AbstractVector}; angular_args...)
    n_ell = length(ells)
    all(f isa Real || length(f) == n_ell for f in (f1_1, f1_2, f2_1, f2_2)) ||
        throw(DimensionMismatch("vector-valued SED factors must have length equal to ells"))
    T_ell = angular_power(corr.shape, ells; angular_args...)
    amplitude = -xi * sqrt(abs(A1 * A2))
    return @. amplitude * (f1_1 * f2_2 + f1_2 * f2_1) * T_ell
end

@inline function correlation_power(corr::TemplateCorrelation{<:TemplateShape},
                                   xi::Real, A1::Real, A2::Real,
                                   f1_1::Real, f1_2::Real, f2_1::Real, f2_2::Real)
    T_ell = angular_power(corr.shape)
    f_cross = f1_1 * f2_2 + f1_2 * f2_1
    factor = -xi * sqrt(abs(A1 * A2)) * f_cross
    return @. factor * T_ell
end

# GeometricMeanCorrelation evaluator from pre-computed autos
@inline function correlation_power(::GeometricMeanCorrelation, ells::AbstractVector,
                                   xi::Real,
                                   D1_11::AbstractVector, D1_22::AbstractVector,
                                   D2_11::AbstractVector, D2_22::AbstractVector)
    n_ell = length(ells)
    all(length(D) == n_ell for D in (D1_11, D1_22, D2_11, D2_22)) ||
        throw(DimensionMismatch("all auto-spectra must have length equal to ells"))
    return @. -xi * (sqrt(abs(D1_11 * D2_22)) + sqrt(abs(D1_22 * D2_11)))
end

# GeometricMeanCorrelation evaluator from component models
function correlation_power(corr::GeometricMeanCorrelation,
                           comp1::SkyComponent, comp2::SkyComponent,
                           ells::AbstractVector, nu1, nu2,
                           xi::Real, A1::Real, A2::Real,
                           comp1_args=(), comp2_args=();
                           comp1_angular_args=NamedTuple(), comp2_angular_args=NamedTuple())
    D1_11 = eval_component(comp1, ells, nu1, nu1, A1, comp1_args...; comp1_angular_args...)
    D1_22 = eval_component(comp1, ells, nu2, nu2, A1, comp1_args...; comp1_angular_args...)
    D2_11 = eval_component(comp2, ells, nu1, nu1, A2, comp2_args...; comp2_angular_args...)
    D2_22 = eval_component(comp2, ells, nu2, nu2, A2, comp2_args...; comp2_angular_args...)
    return correlation_power(corr, ells, xi, D1_11, D1_22, D2_11, D2_22)
end
