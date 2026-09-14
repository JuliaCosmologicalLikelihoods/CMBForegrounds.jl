"""
    bandpass.jl

Bandpass normalization and frequency-integrated SED evaluation.

Mirrors the `_bandpass_construction` and `eval_bandpass` logic in
LAT_MFLike/mflike/foreground.py and fgspectra/frequency.py.

A `Band` is the universal SED-evaluator interface:
- A real passband stores `(nu, norm_bp)` and integrates a SED over ν.
- A `point_band(nu_eff)` is a degenerate δ-band, suitable for surveys
  (SPT, Hillipop) that work with effective frequencies rather than
  actual passband measurements.

All functions are pure (no mutation), compatible with ForwardDiff,
Zygote, and Mooncake autodiff.
"""

# ------------------------------------------------------------------ #
# Trapezoid integration                                                #
# Written explicitly so AD can differentiate through it               #
# ------------------------------------------------------------------ #

"""
    trapz(x, y)

Trapezoidal integration of `y` over `x`.
Both must be `AbstractVector` of the same length.
Pure functional — differentiable via ForwardDiff and Mooncake.
"""
function trapz(x::AbstractVector, y::AbstractVector)
    @assert length(x) == length(y) "trapz: x and y must have the same length"
    s = zero(promote_type(eltype(x), eltype(y)))
    @inbounds for i in 1:(length(x)-1)
        s += (y[i] + y[i+1]) * (x[i+1] - x[i])
    end
    return s / 2
end

# ------------------------------------------------------------------ #
# RawBand — un-normalized passband for bandpass-shift support          #
# ------------------------------------------------------------------ #

"""
    RawBand{T<:Real}

Stores the raw (un-normalized) frequency grid and passband transmission
for one detector array.  Used when bandpass-shift parameters are varied,
so that normalization is recomputed at runtime (differentiably).

Fields:
- `nu`:  frequency grid [GHz]
- `bp`:  raw transmission (proportional to τ(ν)/ν², RJ convention,
         as stored in the SACC file)
"""
struct RawBand{T<:Real}
    nu :: Vector{T}
    bp :: Vector{T}
end

"""
    AbstractBand

Abstract supertype for all bandpass transmission representations.
"""
abstract type AbstractBand end

"""
    DeltaBand{T<:Real} <: AbstractBand

Represents a degenerate Dirac-delta passband at a single effective frequency `nu_eff` (GHz).
"""
struct DeltaBand{T<:Real} <: AbstractBand
    nu_eff::T
end

# ------------------------------------------------------------------ #
# Band struct — holds a normalized passband for one experiment          #
# ------------------------------------------------------------------ #

"""
    Band{T<:Real} <: AbstractBand

Holds the frequency array and normalized transmission for one
experiment/channel.

Fields:
- `nu`:       frequency grid in GHz, length n_freq
- `norm_bp`:  normalized transmission τ̃(ν) = bp·∂B/∂T / ∫ bp·∂B/∂T dν
              (length n_freq; ignored when `monofreq` is true)
- `nu_eff`:   effective (central) frequency in GHz
              (used directly when `monofreq` is true)
- `monofreq`: `true` if this is a Dirac-delta band — no integration

When `monofreq` is `false`, SEDs are integrated over the band.
When `monofreq` is `true`, the SED is evaluated at `nu_eff`.
"""
struct Band{T<:Real} <: AbstractBand
    nu       :: Vector{T}
    norm_bp  :: Vector{T}
    nu_eff   :: T
    monofreq :: Bool
end

"""
    make_band(nu, bp)

Construct a `Band` from raw frequency grid `nu` (GHz) and passband
transmission `bp` (proportional to τ(ν)/ν²; already in the RJ convention
as stored in the SACC file).

The normalization is

    τ̃(ν) = bp(ν) · ∂B_ν/∂T  /  ∫ bp(ν) · ∂B_ν/∂T  dν

with `cmb2bb(ν) ∝ ∂B_ν/∂T`. A length-1 `nu` produces a monochromatic
(Dirac-delta) band.
"""
function make_band(nu::AbstractVector{T}, bp::AbstractVector{T}) where T<:Real
    if length(nu) == 1
        # Monochromatic: Dirac-delta passband, no integration
        return Band{T}(Vector{T}(nu), Vector{T}(bp), nu[1], true)
    end
    w       = bp .* cmb2bb.(nu)
    norm    = trapz(nu, w)
    norm_bp = w ./ norm
    nu_eff  = nu[argmax(bp)]   # approximate center; exact value only for display
    return Band{T}(Vector{T}(nu), norm_bp, T(nu_eff), false)
end

"""
    point_band(nu_eff)

Build a degenerate (Dirac-delta) `Band` representing a single effective
frequency. Useful for surveys (SPT, Hillipop) that pre-bake their
bandpasses into a single ν per channel — they get the same
`integrate_sed`/`eval_sed_bands` API as ACT.
"""
function point_band(nu_eff::T) where T<:Real
    return Band{T}(T[nu_eff], T[one(T)], nu_eff, true)
end

"""
    shift_and_normalize(raw, shift) → Band

Apply a frequency shift `shift` [GHz] to `raw.nu` and return a fully
normalized `Band`.  Differentiable w.r.t. `shift` through `cmb2bb` and `trapz`.
"""
function shift_and_normalize(raw::RawBand{R}, shift::S) where {R<:Real, S<:Real}
    T   = promote_type(R, S)
    nu_s = raw.nu .+ shift
    return make_band(convert(Vector{T}, nu_s), convert(Vector{T}, raw.bp))
end

# ------------------------------------------------------------------ #
# SED evaluation over a band                                           #
# ------------------------------------------------------------------ #

"""
    integrate_sed(sed_fn, band)

Integrate a scalar SED function `sed_fn(ν)` over the normalized bandpass,
returning a single effective SED value:

    ∫ SED(ν) · τ̃(ν) dν   (trapezoidal)

For monochromatic bands, returns `sed_fn(band.nu_eff)` directly.
`sed_fn` must accept a scalar.
"""
function integrate_sed(sed_fn, band::Band{T}) where {T<:Real}
    if band.monofreq
        νmono::T = band.nu[1]
        return sed_fn(νmono)
    end

    ν1::T = band.nu[1]
    y1 = sed_fn(ν1) * band.norm_bp[1]
    y = Vector{typeof(y1)}(undef, length(band.nu))
    y[1] = y1
    @inbounds for i in 2:length(band.nu)
        ν::T = band.nu[i]
        y[i] = sed_fn(ν) * band.norm_bp[i]
    end
    return trapz(band.nu, y)
end

"""
    integrate_tsz(band, nu_0)

Integrate the normalized tSZ SED over a band.
This avoids higher-order closures in hot AD/JET paths.
"""
function integrate_tsz(band::Band{T}, nu_0::S) where {T<:Real,S<:Real}
    if band.monofreq
        return tsz_sed(band.nu[1], nu_0)
    end
    y = tsz_sed(band.nu, nu_0) .* band.norm_bp
    return trapz(band.nu, y)
end

"""
    eval_sed_bands(sed_fn, bands)

Evaluate a SED over an array of `Band`s, returning a vector of length
`n_exp` with one integrated SED value per experiment.

`sed_fn` is a function ν → SED(ν) (scalar → scalar).
"""
function eval_sed_bands(sed_fn, bands::AbstractVector{Band{T}}) where {T<:Real}
    n = length(bands)
    @assert n > 0 "eval_sed_bands: empty band collection"

    b1::Band{T} = bands[1]
    v1 = integrate_sed(sed_fn, b1)
    vals = Vector{typeof(v1)}(undef, n)
    vals[1] = v1
    @inbounds for i in 2:n
        b::Band{T} = bands[i]
        vals[i] = integrate_sed(sed_fn, b)
    end
    return vals
end

@inline integrate_sed(sed_fn, band::DeltaBand) = sed_fn(band.nu_eff)
@inline integrate_tsz(band::DeltaBand, nu_0::Real) = tsz_sed(band.nu_eff, nu_0)

function eval_sed_bands(sed_fn, bands::AbstractVector{<:AbstractBand})
    return [integrate_sed(sed_fn, b) for b in bands]
end

# ------------------------------------------------------------------ #
# Chromatic Beam and Bandpass Integration                              #
# ------------------------------------------------------------------ #

"""
    ChromaticBeam{L<:AbstractVector, M<:AbstractMatrix}

Represents a frequency-dependent beam window function ``b_\\ell(\\nu)``.

# Fields
- `ells::L`: multipole grid (length `n_ell`)
- `beam::M`: 2D matrix of beam window functions of shape `(n_ell, n_nu)`,
  where `n_nu` corresponds to the frequency grid `band.nu`.
"""
struct ChromaticBeam{L<:AbstractVector, M<:AbstractMatrix}
    ells::L
    beam::M
    function ChromaticBeam(ells::L, beam::M) where {L<:AbstractVector, M<:AbstractMatrix}
        @assert length(ells) == size(beam, 1) "ChromaticBeam: ells length must match beam first dimension"
        return new{L, M}(ells, beam)
    end
end

"""
    integrate_chromatic_sed(sed_fn, band::Band, chromatic_beam::ChromaticBeam) -> Vector

Evaluate the multipole-dependent chromatic band response ``F_\\ell`` (length `n_ell`):
```math
F_\\ell = \\frac{\\int d\\nu \\, b_\\ell(\\nu) \\, \\mathrm{bp}(\\nu) \\, \\mathrm{cmb2bb}(\\nu) \\, f(\\nu)}{\\int d\\nu \\, b_\\ell(\\nu) \\, \\mathrm{bp}(\\nu) \\, \\mathrm{cmb2bb}(\\nu)}
```
Matches Eq. (31) of the ACT DR6 paper (Louis et al. 2025).

# Limits:
1. **Delta limit**: For monochromatic band, returns `fill(sed_fn(band.nu_eff), n_ell)`.
2. **Achromatic limit**: When ``b_\\ell(\\nu) = b_\\ell`` is independent of frequency,
   the beam cancels between numerator and denominator, returning `fill(integrate_sed(sed_fn, band), n_ell)`.
3. **Normalized blackbody limit**: For ``f(\\nu) \\equiv 1`` (CMB/kSZ), numerator equals denominator,
   returning `ones(n_ell)`.
"""
function integrate_chromatic_sed(sed_fn, band::Band{T}, chromatic_beam::ChromaticBeam) where {T<:Real}
    n_ell = length(chromatic_beam.ells)
    if band.monofreq
        val = sed_fn(band.nu[1])
        return fill(val, n_ell)
    end

    n_nu = length(band.nu)
    @assert size(chromatic_beam.beam, 2) == n_nu "ChromaticBeam: beam second dimension must match band frequency count"

    # Pre-evaluate SED across frequency grid
    f1 = sed_fn(band.nu[1])
    sed_vals = Vector{typeof(f1)}(undef, n_nu)
    sed_vals[1] = f1
    @inbounds for i in 2:n_nu
        sed_vals[i] = sed_fn(band.nu[i])
    end

    R = promote_type(T, typeof(f1), eltype(chromatic_beam.beam))
    F_ell = Vector{R}(undef, n_ell)

    y_num = Vector{R}(undef, n_nu)
    y_den = Vector{R}(undef, n_nu)

    @inbounds for ℓ in 1:n_ell
        for i in 1:n_nu
            b_val = chromatic_beam.beam[ℓ, i]
            w = b_val * band.norm_bp[i]
            y_den[i] = w
            y_num[i] = w * sed_vals[i]
        end
        num = trapz(band.nu, y_num)
        den = trapz(band.nu, y_den)
        F_ell[ℓ] = num / den
    end

    return F_ell
end

@inline integrate_chromatic_sed(sed_fn, band::DeltaBand, chromatic_beam::ChromaticBeam) = fill(sed_fn(band.nu_eff), length(chromatic_beam.ells))

"""
    eval_chromatic_sed_bands(sed_fn, bands, chromatic_beams) -> Matrix

Evaluate chromatic SED weights across an array of `Band`s and their corresponding `ChromaticBeam`s.
Returns a 2D matrix of shape `(n_freq, n_ell)`.
"""
function eval_chromatic_sed_bands(sed_fn, bands::AbstractVector{<:AbstractBand}, chromatic_beams::AbstractVector{<:ChromaticBeam})
    n_freq = length(bands)
    @assert length(chromatic_beams) == n_freq "Number of bands and chromatic beams must match"
    n_ell = length(chromatic_beams[1].ells)

    F1 = integrate_chromatic_sed(sed_fn, bands[1], chromatic_beams[1])
    F = Matrix{eltype(F1)}(undef, n_freq, n_ell)
    F[1, :] = F1
    @inbounds for i in 2:n_freq
        F[i, :] = integrate_chromatic_sed(sed_fn, bands[i], chromatic_beams[i])
    end
    return F
end
