"""
    sed.jl

Spectral Energy Distribution (SED) models and multiple-dispatch interface.

Provides mathematical representations for the frequency scaling of CMB foregrounds:
- `ModifiedBlackbodySED`: modified blackbody emission (Galactic dust, clustered and Poisson CIB)
- `RadioSED`: power law in frequency with explicit RJ vs flux-density convention
- `ThermalSZSED`: non-relativistic thermal Sunyaev-Zel'dovich spectral response
- `ConstantSED`: frequency-independent response in K_CMB (kSZ)
- `NoSED`: identity / unit SED for empirical residuals (e.g. CamSpec TT power laws)

Evaluated through `sed_weight` and composed with angular shapes in `eval_component`.
"""

abstract type AbstractSED end

"""
    ModifiedBlackbodySED{T1<:Real, T2<:Real, T3<:Real} <: AbstractSED

Modified blackbody (MBB) spectral energy distribution:
```math
S(\\nu) = \\left(\\frac{\\nu}{\\nu_0}\\right)^\\beta
          \\frac{B(\\nu, T_\\mathrm{dust})}{B(\\nu_0, T_\\mathrm{dust})}
          \\frac{(\\partial B/\\partial T)(\\nu_0, T_\\mathrm{CMB})}{(\\partial B/\\partial T)(\\nu, T_\\mathrm{CMB})}
```

# Fields
- `nu_0::T1`: Reference frequency in GHz.
- `T_dust::T2`: Dust temperature in Kelvin.
- `T_CMB::T3`: CMB temperature in Kelvin (default `CMBForegrounds.T_CMB`).
"""
struct ModifiedBlackbodySED{T1<:Real, T2<:Real, T3<:Real} <: AbstractSED
    nu_0::T1
    T_dust::T2
    T_CMB::T3
end

function ModifiedBlackbodySED(nu_0::Real, T_dust::Real; T_CMB::Real=T_CMB)
    n0, Td, Tc = promote(float(nu_0), float(T_dust), float(T_CMB))
    return ModifiedBlackbodySED(n0, Td, Tc)
end

"""
    RadioSED{T1<:Real, T2<:Real} <: AbstractSED

Power-law radio point-source spectral energy distribution.

# Radio spectral index conventions:
- `:rj` (default): Rayleigh-Jeans brightness temperature index ``\\beta_\\mathrm{RJ}``
  (typically ``\\approx -2.7``). Model:
  ```math
  S(\\nu) = \\left(\\frac{\\nu}{\\nu_0}\\right)^{\\beta_\\mathrm{RJ}}
            \\frac{\\mathrm{rj2cmb}(\\nu)}{\\mathrm{rj2cmb}(\\nu_0)}
  ```
  Matches [`radio_sed`](@ref) and ACT DR6 convention.
- `:flux`: Flux-density spectral index ``\\beta_\\mathrm{flux}`` (``S_\\nu \\propto \\nu^{\\beta_\\mathrm{flux}}``,
  typically ``\\approx -0.7``). Model:
  ```math
  S(\\nu) = \\left(\\frac{\\nu}{\\nu_0}\\right)^{\\beta_\\mathrm{flux}}
            \\Big/ \\frac{(\\partial B/\\partial T)(\\nu)}{(\\partial B/\\partial T)(\\nu_0)}
  ```
  Matches [`_radio_sed_ratio`](@ref) and Planck/HiLLiPoP/SPT convention.

The two conventions are related by:
```math
\\beta_\\mathrm{RJ} = \\beta_\\mathrm{flux} - 2
```

# Fields
- `nu_0::T1`: Reference frequency in GHz.
- `convention::Symbol`: `:rj` or `:flux`.
- `T_CMB::T2`: CMB temperature in Kelvin.
"""
struct RadioSED{T1<:Real, T2<:Real} <: AbstractSED
    nu_0::T1
    convention::Symbol
    T_CMB::T2

    function RadioSED(nu_0::Real; convention::Symbol=:rj, T_CMB::Real=T_CMB)
        @assert convention in (:rj, :flux) "RadioSED convention must be :rj or :flux"
        n0, Tc = promote(float(nu_0), float(T_CMB))
        return new{typeof(n0), typeof(Tc)}(n0, convention, Tc)
    end
end

"""
    ThermalSZSED{T1<:Real, T2<:Real} <: AbstractSED

Thermal Sunyaev-Zel'dovich (tSZ) non-relativistic spectral energy distribution:
```math
S(\\nu) = \\frac{g(\\nu)}{g(\\nu_0)}
```
where ``g(x) = x \\coth(x/2) - 4`` with ``x = h\\nu/(k_B T_\\mathrm{CMB})``.

# Fields
- `nu_0::T1`: Reference frequency in GHz.
- `T_CMB::T2`: CMB temperature in Kelvin.
"""
struct ThermalSZSED{T1<:Real, T2<:Real} <: AbstractSED
    nu_0::T1
    T_CMB::T2
end

function ThermalSZSED(nu_0::Real; T_CMB::Real=T_CMB)
    n0, Tc = promote(float(nu_0), float(T_CMB))
    return ThermalSZSED(n0, Tc)
end

"""
    ConstantSED <: AbstractSED

Frequency-independent unit SED in thermodynamic K_CMB units:
```math
S(\\nu) = 1
```
Used for kinematic SZ (kSZ), which has a pure blackbody spectrum.
"""
struct ConstantSED <: AbstractSED end

"""
    NoSED <: AbstractSED

Identity / no-op SED returning unit response for any frequency.
Used for empirical residuals (e.g. CamSpec TT power-law residuals) that do not
possess a physical frequency scaling.
"""
struct NoSED <: AbstractSED end

"""
    SkyComponent{S<:AbstractSED, A<:AbstractAngularModel}

Composition of an SED frequency scaling and an angular multipole model.
"""
struct SkyComponent{S<:AbstractSED, A<:AbstractAngularModel}
    sed::S
    angular::A
end

# ------------------------------------------------------------------ #
# Common evaluation interface: sed_weight                              #
# ------------------------------------------------------------------ #

"""
    sed_weight(sed::AbstractSED, nu_or_band, args...)

Evaluate the dimensionless SED weight for a given frequency or passband.
"""
function sed_weight end

# ModifiedBlackbodySED
@inline function sed_weight(sed::ModifiedBlackbodySED, nu::Real, beta::Real)
    return cib_mbb_sed_weight(beta, sed.T_dust, sed.nu_0, nu; T_CMB=sed.T_CMB)
end

@inline function sed_weight(sed::ModifiedBlackbodySED, nu::AbstractVector{<:Real}, beta::Real)
    return cib_mbb_sed_weight.(beta, sed.T_dust, sed.nu_0, nu; T_CMB=sed.T_CMB)
end

@inline function sed_weight(sed::ModifiedBlackbodySED, band::Band, beta::Real)
    if band.monofreq
        return sed_weight(sed, band.nu[1], beta)
    end
    fn = ν -> sed_weight(sed, ν, beta)
    return integrate_sed(fn, band)
end

# RadioSED
@inline function sed_weight(sed::RadioSED, nu::Real, beta::Real)
    if sed.convention === :flux
        return _radio_sed_ratio(nu, sed.nu_0, beta, sed.T_CMB)
    else
        return radio_sed(nu, sed.nu_0, beta)
    end
end

@inline function sed_weight(sed::RadioSED, nu::AbstractVector{<:Real}, beta::Real)
    if sed.convention === :flux
        return _radio_sed_ratio.(nu, sed.nu_0, beta, sed.T_CMB)
    else
        return radio_sed(nu, sed.nu_0, beta)
    end
end

@inline function sed_weight(sed::RadioSED, band::Band, beta::Real)
    if band.monofreq
        return sed_weight(sed, band.nu[1], beta)
    end
    fn = ν -> sed_weight(sed, ν, beta)
    return integrate_sed(fn, band)
end

# ThermalSZSED
@inline function sed_weight(sed::ThermalSZSED, nu::Real)
    return tsz_sed(nu, sed.nu_0)
end

@inline function sed_weight(sed::ThermalSZSED, nu::AbstractVector{<:Real})
    return tsz_sed(nu, sed.nu_0)
end

@inline function sed_weight(sed::ThermalSZSED, band::Band)
    return integrate_tsz(band, sed.nu_0)
end

# DeltaBand generic fallback
@inline sed_weight(sed::AbstractSED, band::DeltaBand, args...) = sed_weight(sed, band.nu_eff, args...)

# ConstantSED
@inline sed_weight(::ConstantSED, ::Union{Real, AbstractBand}) = 1.0
@inline sed_weight(::ConstantSED, nu::AbstractVector{<:Real}) = ones(eltype(nu), length(nu))
@inline sed_weight(::ConstantSED, ::AbstractBand, chromatic_beam::ChromaticBeam) = ones(length(chromatic_beam.ells))

# NoSED
@inline sed_weight(::NoSED, ::Union{Real, AbstractBand}) = 1.0
@inline sed_weight(::NoSED, nu::AbstractVector{<:Real}) = ones(eltype(nu), length(nu))
@inline sed_weight(::NoSED, ::AbstractBand, chromatic_beam::ChromaticBeam) = ones(length(chromatic_beam.ells))

# Chromatic beam single-band evaluation
@inline function sed_weight(sed::AbstractSED, band::AbstractBand, chromatic_beam::ChromaticBeam, args...)
    fn = ν -> sed_weight(sed, ν, args...)
    return integrate_chromatic_sed(fn, band, chromatic_beam)
end

# Collection of Bands helper
@inline function sed_weight(sed::AbstractSED, bands::AbstractVector{<:AbstractBand}, args...)
    return [sed_weight(sed, b, args...) for b in bands]
end

# Collection of Bands with ChromaticBeams helper
@inline function sed_weight(sed::AbstractSED, bands::AbstractVector{<:AbstractBand}, chromatic_beams::AbstractVector{<:ChromaticBeam}, args...)
    fn = ν -> sed_weight(sed, ν, args...)
    return eval_chromatic_sed_bands(fn, bands, chromatic_beams)
end

# ------------------------------------------------------------------ #
# Foreground component composition helpers                            #
# ------------------------------------------------------------------ #

"""
    eval_component(sed::AbstractSED, angular::AbstractAngularModel, ells, nu1, nu2, amp, sed_args...; angular_args...)

Evaluate a single cross-spectrum ``D_\\ell(\\nu_1, \\nu_2)`` for a physical component:
```math
D_\\ell = \\mathrm{amp} \\cdot S(\\nu_1) \\cdot S(\\nu_2) \\cdot D_\\ell^\\mathrm{ang}
```
"""
function eval_component(sed::AbstractSED, angular::AbstractAngularModel, ells::AbstractVector,
                        nu1, nu2, amp::Real, sed_args...; angular_args...)
    s1 = sed_weight(sed, nu1, sed_args...)
    s2 = sed_weight(sed, nu2, sed_args...)
    ang = angular_power(angular, ells, values(angular_args)...; amp=amp)
    return @. (s1 * s2) * ang
end

function eval_component(comp::SkyComponent, ells::AbstractVector,
                        nu1, nu2, amp::Real, sed_args...; angular_args...)
    return eval_component(comp.sed, comp.angular, ells, nu1, nu2, amp, sed_args...; angular_args...)
end

"""
    eval_component(sed1::AbstractSED, sed2::AbstractSED, angular::AbstractAngularModel, ells, nu1, nu2, amp, sed1_args=(), sed2_args=(); angular_args...)

Evaluate a cross-spectrum with different SED properties on leg 1 and leg 2 (e.g. TE or distinct map emissivities).
"""
function eval_component(sed1::AbstractSED, sed2::AbstractSED, angular::AbstractAngularModel, ells::AbstractVector,
                        nu1, nu2, amp::Real, sed1_args::Tuple=(), sed2_args::Tuple=(); angular_args...)
    s1 = sed_weight(sed1, nu1, sed1_args...)
    s2 = sed_weight(sed2, nu2, sed2_args...)
    ang = angular_power(angular, ells, values(angular_args)...; amp=amp)
    return @. (s1 * s2) * ang
end

"""
    eval_component(sed::AbstractSED, angular::AbstractAngularModel, ells, bands, amp, sed_args...; angular_args...)

Evaluate full frequency-cross 3D tensor `(n_freq, n_freq, n_ell)` for an array of bands.
"""
function eval_component(sed::AbstractSED, angular::AbstractAngularModel, ells::AbstractVector,
                        bands::AbstractVector{<:AbstractBand}, amp::Real, sed_args...; angular_args...)
    f = [sed_weight(sed, b, sed_args...) for b in bands]
    cl = angular_power(angular, ells, values(angular_args)...; amp=amp)
    return factorized_cross(f, cl)
end

"""
    eval_component(sed::AbstractSED, angular::AbstractAngularModel, ells, bands, chromatic_beams, amp, sed_args...; angular_args...)

Evaluate full frequency-cross 3D tensor `(n_freq, n_freq, n_ell)` with chromatic beams.
"""
function eval_component(sed::AbstractSED, angular::AbstractAngularModel, ells::AbstractVector,
                        bands::AbstractVector{<:AbstractBand}, chromatic_beams::AbstractVector{<:ChromaticBeam},
                        amp::Real, sed_args...; angular_args...)
    F = eval_chromatic_sed_bands(ν -> sed_weight(sed, ν, sed_args...), bands, chromatic_beams)
    cl = angular_power(angular, ells, values(angular_args)...; amp=amp)
    return factorized_cross(F, cl)
end

function eval_component(comp::SkyComponent, ells::AbstractVector,
                        bands::AbstractVector{<:AbstractBand}, chromatic_beams::AbstractVector{<:ChromaticBeam},
                        amp::Real, sed_args...; angular_args...)
    return eval_component(comp.sed, comp.angular, ells, bands, chromatic_beams, amp, sed_args...; angular_args...)
end

"""
    eval_component_te(sedT::AbstractSED, sedE::AbstractSED, angular::AbstractAngularModel, ells, bandsT, bandsE, amp, sedT_args=(), sedE_args=(); angular_args...)

Evaluate full frequency-cross 3D tensor `(n_freq, n_freq, n_ell)` for TE polarization cross.
"""
function eval_component_te(sedT::AbstractSED, sedE::AbstractSED, angular::AbstractAngularModel, ells::AbstractVector,
                           bandsT::AbstractVector{<:AbstractBand}, bandsE::AbstractVector{<:AbstractBand}, amp::Real,
                           sedT_args::Tuple=(), sedE_args::Tuple=(); angular_args...)
    fT = [sed_weight(sedT, b, sedT_args...) for b in bandsT]
    fE = [sed_weight(sedE, b, sedE_args...) for b in bandsE]
    cl = angular_power(angular, ells, values(angular_args)...; amp=amp)
    return factorized_cross_te(fT, fE, cl)
end

"""
    eval_component_te(sedT::AbstractSED, sedE::AbstractSED, angular::AbstractAngularModel, ells, bandsT, beamsT, bandsE, beamsE, amp, sedT_args=(), sedE_args=(); angular_args...)

Evaluate full frequency-cross 3D tensor `(n_freq, n_freq, n_ell)` for TE polarization cross with chromatic beams.
"""
function eval_component_te(sedT::AbstractSED, sedE::AbstractSED, angular::AbstractAngularModel, ells::AbstractVector,
                           bandsT::AbstractVector{<:AbstractBand}, beamsT::AbstractVector{<:ChromaticBeam},
                           bandsE::AbstractVector{<:AbstractBand}, beamsE::AbstractVector{<:ChromaticBeam},
                           amp::Real, sedT_args::Tuple=(), sedE_args::Tuple=(); angular_args...)
    FT = eval_chromatic_sed_bands(ν -> sed_weight(sedT, ν, sedT_args...), bandsT, beamsT)
    FE = eval_chromatic_sed_bands(ν -> sed_weight(sedE, ν, sedE_args...), bandsE, beamsE)
    cl = angular_power(angular, ells, values(angular_args)...; amp=amp)
    return factorized_cross_te(FT, FE, cl)
end
