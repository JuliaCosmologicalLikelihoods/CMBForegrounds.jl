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

"""
    AbstractSED

Abstract supertype for all spectral energy distribution representations.
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
  Matches `radio_sed` and ACT DR6 convention.
- `:flux`: Flux-density spectral index ``\\beta_\\mathrm{flux}`` (``S_\\nu \\propto \\nu^{\\beta_\\mathrm{flux}}``,
  typically ``\\approx -0.7``). Model:
  ```math
  S(\\nu) = \\left(\\frac{\\nu}{\\nu_0}\\right)^{\\beta_\\mathrm{flux}}
            \\Big/ \\frac{(\\partial B/\\partial T)(\\nu)}{(\\partial B/\\partial T)(\\nu_0)}
  ```
  Matches `_radio_sed_ratio` and Planck/HiLLiPoP/SPT convention.

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
Defining scalar-frequency evaluation for a custom `AbstractSED` automatically
provides `DeltaBand`, tabulated `Band`, and chromatic-band lifting.
"""
function sed_weight end

# ModifiedBlackbodySED
@inline function sed_weight(sed::ModifiedBlackbodySED, nu::Real, beta::Real)
    return cib_mbb_sed_weight(beta, sed.T_dust, sed.nu_0, nu; T_CMB=sed.T_CMB)
end

# RadioSED
@inline function sed_weight(sed::RadioSED, nu::Real, beta::Real)
    if sed.convention === :flux
        return _radio_sed_ratio(nu, sed.nu_0, beta, sed.T_CMB)
    else
        return radio_sed(nu, sed.nu_0, beta, sed.T_CMB)
    end
end

# ThermalSZSED
@inline function sed_weight(sed::ThermalSZSED, nu::Real)
    return tsz_sed(nu, sed.nu_0, sed.T_CMB)
end

@inline function sed_weight(sed::ThermalSZSED, band::Band)
    return integrate_tsz(band, sed.nu_0, sed.T_CMB)
end

# DeltaBand generic fallback
@inline sed_weight(sed::AbstractSED, band::DeltaBand, args...) = sed_weight(sed, band.nu_eff, args...)
@inline function sed_weight(sed::AbstractSED, band::DeltaBand,
                            chromatic_beam::ChromaticBeam, args...)
    return fill(sed_weight(sed, band.nu_eff, args...), length(chromatic_beam.ells))
end

# ConstantSED
@inline sed_weight(::ConstantSED, nu::Real) = one(nu)
# NoSED
@inline sed_weight(::NoSED, nu::Real) = one(nu)

@inline function sed_weight(sed::AbstractSED, nu::AbstractVector{<:Real}, args...)
    return map(value -> sed_weight(sed, value, args...), nu)
end

# Generic scalar-SED lifting to a tabulated band.
@inline function sed_weight(sed::AbstractSED, band::Band, args...)
    return integrate_sed(ν -> sed_weight(sed, ν, args...), band)
end

@inline function sed_weight(sed::AbstractSED, band::Band,
                            chromatic_beam::ChromaticBeam, args...)
    return integrate_chromatic_sed(ν -> sed_weight(sed, ν, args...), band, chromatic_beam)
end

@inline function sed_weight(sed::AbstractSED,
                            prepared::PreparedChromaticBandpass, args...)
    return integrate_chromatic_sed(ν -> sed_weight(sed, ν, args...), prepared)
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


@inline function sed_weight(
    sed::AbstractSED,
    prepared::AbstractVector{<:PreparedChromaticBandpass}, args...
)
    return eval_chromatic_sed_bands(ν -> sed_weight(sed, ν, args...), prepared)
end

# ------------------------------------------------------------------ #
# Foreground component composition helpers                            #
# ------------------------------------------------------------------ #

"""
    eval_component(sed::AbstractSED, angular::AbstractAngularModel, ells, nu1, nu2, amp, sed_args...; angular_args...)

Evaluate a single cross-spectrum ``D_\\ell(\\nu_1, \\nu_2)`` for a physical component.
Each frequency leg may be a scalar frequency, a vector of frequencies, an
`AbstractBand`, or a `PreparedChromaticBandpass`:
```math
D_\\ell = \\mathrm{amp} \\cdot S(\\nu_1) \\cdot S(\\nu_2) \\cdot D_\\ell^\\mathrm{ang}
```
"""
function eval_component(sed::AbstractSED, angular::AbstractAngularModel,
                        ells::AbstractVector,
                        nu1::Union{Real, AbstractVector{<:Real}, AbstractBand,
                                   PreparedChromaticBandpass},
                        nu2::Union{Real, AbstractVector{<:Real}, AbstractBand,
                                   PreparedChromaticBandpass},
                        amp::Real, sed_args...; angular_args...)
    return eval_component(sed, sed, angular, ells, nu1, nu2, amp,
                          sed_args, sed_args; angular_args...)
end

@inline _validate_spectrum_weight(::Real, ::Integer) = nothing

@inline function _validate_spectrum_weight(weight::AbstractVector, n_ell::Integer)
    length(weight) == n_ell ||
        throw(DimensionMismatch("vector-valued SED weights must have length equal to ells"))
    return nothing
end

@inline function _combine_component_weights(s1, s2, angular::AbstractAngularModel,
                                            ells::AbstractVector, amp::Real;
                                            angular_args...)
    n_ell = length(ells)
    _validate_spectrum_weight(s1, n_ell)
    _validate_spectrum_weight(s2, n_ell)
    shape = angular_power(angular, ells; amp=amp, angular_args...)
    return @. s1 * s2 * shape
end

"""
    eval_component(sed1::AbstractSED, sed2::AbstractSED, angular::AbstractAngularModel, ells, nu1, nu2, amp, sed1_args=(), sed2_args=(); angular_args...)

Evaluate a cross-spectrum with different SED properties on leg 1 and leg 2 (e.g. TE or distinct map emissivities).
"""
function eval_component(sed1::AbstractSED, sed2::AbstractSED, angular::AbstractAngularModel, ells::AbstractVector,
                        nu1::Union{Real, AbstractVector{<:Real}, AbstractBand,
                                   PreparedChromaticBandpass},
                        nu2::Union{Real, AbstractVector{<:Real}, AbstractBand,
                                   PreparedChromaticBandpass},
                        amp::Real, sed1_args::Tuple=(), sed2_args::Tuple=(); angular_args...)
    s1 = sed_weight(sed1, nu1, sed1_args...)
    s2 = sed_weight(sed2, nu2, sed2_args...)
    return _combine_component_weights(s1, s2, angular, ells, amp; angular_args...)
end

@inline eval_component(comp::SkyComponent, args...; kwargs...) =
    eval_component(comp.sed, comp.angular, args...; kwargs...)

"""
    eval_component(sed::AbstractSED, angular::AbstractAngularModel, ells, bands, amp, sed_args...; angular_args...)

Evaluate full frequency-cross 3D tensor `(n_freq, n_freq, n_ell)` for an array of bands.
"""
function eval_component(sed::AbstractSED, angular::AbstractAngularModel, ells::AbstractVector,
                        bands::AbstractVector{<:AbstractBand}, amp::Real, sed_args...; angular_args...)
    f = [sed_weight(sed, b, sed_args...) for b in bands]
    cl = angular_power(angular, ells; amp=amp, angular_args...)
    return factorized_cross(f, cl)
end

"""
    eval_component(sed::AbstractSED, angular::AbstractAngularModel, ells, bands, chromatic_beams, amp, sed_args...; angular_args...)

Evaluate full frequency-cross 3D tensor `(n_freq, n_freq, n_ell)` with chromatic beams.
"""
function eval_component(sed::AbstractSED, angular::AbstractAngularModel, ells::AbstractVector,
                        bands::AbstractVector{<:AbstractBand}, chromatic_beams::AbstractVector{<:ChromaticBeam},
                        amp::Real, sed_args...; angular_args...)
    all(beam -> _same_multipole_grid(beam.ells, ells), chromatic_beams) ||
        throw(ArgumentError("chromatic beam multipoles must match ells"))
    F = eval_chromatic_sed_bands(ν -> sed_weight(sed, ν, sed_args...), bands, chromatic_beams)
    cl = angular_power(angular, ells; amp=amp, angular_args...)
    return factorized_cross(F, cl)
end

function eval_component(
    sed::AbstractSED, angular::AbstractAngularModel, ells::AbstractVector,
    prepared::AbstractVector{<:PreparedChromaticBandpass}, amp::Real,
    sed_args...; angular_args...
)
    all(response -> _same_multipole_grid(response.beam.ells, ells), prepared) ||
        throw(ArgumentError("prepared response multipoles must match ells"))
    F = sed_weight(sed, prepared, sed_args...)
    cl = angular_power(angular, ells; amp=amp, angular_args...)
    return factorized_cross(F, cl)
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
    cl = angular_power(angular, ells; amp=amp, angular_args...)
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
    all(beam -> _same_multipole_grid(beam.ells, ells), beamsT) ||
        throw(ArgumentError("temperature chromatic beam multipoles must match ells"))
    all(beam -> _same_multipole_grid(beam.ells, ells), beamsE) ||
        throw(ArgumentError("polarization chromatic beam multipoles must match ells"))
    FT = eval_chromatic_sed_bands(ν -> sed_weight(sedT, ν, sedT_args...), bandsT, beamsT)
    FE = eval_chromatic_sed_bands(ν -> sed_weight(sedE, ν, sedE_args...), bandsE, beamsE)
    cl = angular_power(angular, ells; amp=amp, angular_args...)
    return factorized_cross_te(FT, FE, cl)
end

function eval_component_te(
    sedT::AbstractSED, sedE::AbstractSED, angular::AbstractAngularModel,
    ells::AbstractVector,
    preparedT::AbstractVector{<:PreparedChromaticBandpass},
    preparedE::AbstractVector{<:PreparedChromaticBandpass}, amp::Real,
    sedT_args::Tuple=(), sedE_args::Tuple=(); angular_args...
)
    all(response -> _same_multipole_grid(response.beam.ells, ells), preparedT) ||
        throw(ArgumentError("temperature prepared response multipoles must match ells"))
    all(response -> _same_multipole_grid(response.beam.ells, ells), preparedE) ||
        throw(ArgumentError("polarization prepared response multipoles must match ells"))
    FT = sed_weight(sedT, preparedT, sedT_args...)
    FE = sed_weight(sedE, preparedE, sedE_args...)
    cl = angular_power(angular, ells; amp=amp, angular_args...)
    return factorized_cross_te(FT, FE, cl)
end
