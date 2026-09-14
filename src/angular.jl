"""
    angular.jl

Angular power spectrum models and multiple-dispatch interface.

Provides concrete mathematical representations for the angular (multipole ℓ)
dependence of CMB foregrounds and empirical residuals:
- `PowerLawShape`: D_ℓ ∝ (ℓ/ℓ₀)^α
- `PoissonShape`: exact constant C_ℓ, D_ℓ ∝ ℓ(ℓ+1) / [ℓ₀(ℓ₀+1)]
- `TemplateShape`: tabulated D_ℓ template, with optional normalization and arbitrary ℓ_min
- `TiltedTemplateShape`: template rescaled by a power-law tilt (ℓ/ℓ₀)^α

All representations evaluate through the unified `angular_power` interface.
"""

abstract type AbstractAngularModel end

"""
    PowerLawShape{T<:Real} <: AbstractAngularModel

Power-law angular dependence:
```math
D_\\ell = \\mathrm{amp} \\cdot \\left(\\frac{\\ell}{\\ell_0}\\right)^\\alpha
```

# Fields
- `ell_0::T`: Reference/pivot multipole ℓ₀.
"""
struct PowerLawShape{T<:Real} <: AbstractAngularModel
    ell_0::T
end

PowerLawShape() = PowerLawShape(3000.0)

"""
    PoissonShape{T<:Real} <: AbstractAngularModel

Exact constant-``C_\\ell`` Poisson (shot-noise) angular shape:
```math
D_\\ell = \\mathrm{amp} \\cdot \\frac{\\ell(\\ell+1)}{\\ell_0(\\ell_0+1)}
```

This represents the exact ``C_\\ell = \\mathrm{const}`` scaling, distinct from the
``\\ell^2`` power-law approximation.

# Fields
- `ell_0::T`: Reference/pivot multipole ℓ₀ (default 3000).
"""
struct PoissonShape{T<:Real} <: AbstractAngularModel
    ell_0::T
end

PoissonShape() = PoissonShape(3000.0)

"""
    TemplateShape{V<:AbstractVector, T<:Union{Nothing, Real}, I<:Integer} <: AbstractAngularModel

Tabulated ``D_\\ell`` template.

# Fields
- `template::V`: Array of template values.
- `ell_0::T`: Pivot multipole where the template is normalized to 1, or `nothing` if already normalized.
- `ell_min::I`: Starting multipole corresponding to `template[1]` (default 0).
"""
struct TemplateShape{V<:AbstractVector, T<:Union{Nothing, Real}, I<:Integer} <: AbstractAngularModel
    template::V
    ell_0::T
    ell_min::I
end

function TemplateShape(template::AbstractVector, ell_0::Union{Nothing, Real}; ell_min::Integer=0)
    return TemplateShape{typeof(template), typeof(ell_0), typeof(ell_min)}(template, ell_0, ell_min)
end

function TemplateShape(template::AbstractVector; ell_0::Union{Nothing, Real}=3000.0, ell_min::Integer=0)
    return TemplateShape(template, ell_0; ell_min=ell_min)
end

"""
    TiltedTemplateShape{S<:AbstractAngularModel, T<:Real} <: AbstractAngularModel

Tabulated template rescaled by a power-law tilt:
```math
D_\\ell = \\mathrm{amp} \\cdot T_\\ell \\cdot \\left(\\frac{\\ell}{\\ell_0}\\right)^\\alpha
```

# Fields
- `shape::S`: Base angular model (typically a `TemplateShape`).
- `ell_0::T`: Pivot multipole for the tilt factor (default 3000).
"""
struct TiltedTemplateShape{S<:AbstractAngularModel, T<:Real} <: AbstractAngularModel
    shape::S
    ell_0::T
end

TiltedTemplateShape(shape::AbstractAngularModel) = TiltedTemplateShape(shape, 3000.0)

function TiltedTemplateShape(template::AbstractVector, ell_0::Real; ell_min::Integer=0)
    base = TemplateShape(template; ell_0=nothing, ell_min=ell_min)
    return TiltedTemplateShape(base, ell_0)
end

function TiltedTemplateShape(template::AbstractVector; ell_0::Real=3000.0, ell_min::Integer=0)
    return TiltedTemplateShape(template, ell_0; ell_min=ell_min)
end

# ------------------------------------------------------------------ #
# Common evaluation interface: angular_power                           #
# ------------------------------------------------------------------ #

"""
    angular_power(model::AbstractAngularModel, ell, args...; amp=1.0)

Evaluate the angular power spectrum shape ``D_\\ell`` for the given representation.
"""
function angular_power end

@inline function angular_power(model::PowerLawShape, ell::AbstractVector, alpha::Real; amp::Real=1.0)
    return eval_powerlaw(ell, model.ell_0, alpha; amp=amp)
end

@inline function angular_power(model::PowerLawShape, ell::AbstractVector; amp::Real=1.0, alpha::Real=0.0)
    return eval_powerlaw(ell, model.ell_0, alpha; amp=amp)
end

@inline function angular_power(model::PoissonShape, ell::AbstractVector; amp::Real=1.0)
    norm = model.ell_0 * (model.ell_0 + 1)
    return @. amp * (ell * (ell + 1)) / norm
end

@inline function angular_power(model::PoissonShape, ell::AbstractVector, alpha::Real; amp::Real=1.0)
    return angular_power(model, ell; amp=amp)
end

@inline function _template_val(model::TemplateShape, ell::AbstractVector)
    if length(ell) == length(model.template)
        if model.ell_0 === nothing
            return model.template
        else
            idx0 = findfirst(isequal(model.ell_0), ell)
            norm = idx0 !== nothing ? model.template[idx0] : one(eltype(model.template))
            return norm == 1 ? model.template : model.template ./ norm
        end
    end
    norm = model.ell_0 === nothing ? one(eltype(model.template)) : model.template[round(Int, model.ell_0) - model.ell_min + 1]
    idx = @. round(Int, ell) - model.ell_min + 1
    vals = model.template[idx]
    if norm == 1
        return vals
    else
        return vals ./ norm
    end
end

@inline function angular_power(model::TemplateShape, ell::AbstractVector; amp::Real=1.0)
    t = _template_val(model, ell)
    return @. amp * t
end

@inline function angular_power(model::TemplateShape, ell::AbstractVector, alpha::Real; amp::Real=1.0)
    return angular_power(model, ell; amp=amp)
end

@inline function angular_power(model::TemplateShape; amp::Real=1.0)
    norm = model.ell_0 === nothing ? one(eltype(model.template)) : model.template[Int(model.ell_0) - model.ell_min + 1]
    if norm == 1
        return @. amp * model.template
    else
        return @. amp * (model.template / norm)
    end
end

@inline function angular_power(model::TiltedTemplateShape, ell::AbstractVector, alpha::Real; amp::Real=1.0)
    base = angular_power(model.shape, ell; amp=amp)
    return @. base * (ell / model.ell_0)^alpha
end

@inline function angular_power(model::TiltedTemplateShape, ell::AbstractVector; amp::Real=1.0, alpha::Real=0.0)
    base = angular_power(model.shape, ell; amp=amp)
    if iszero(alpha)
        return base
    end
    return @. base * (ell / model.ell_0)^alpha
end
