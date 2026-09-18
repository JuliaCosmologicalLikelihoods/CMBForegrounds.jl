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

function _validate_band_inputs(nu::AbstractVector, bp::AbstractVector)
    Base.require_one_based_indexing(nu, bp)
    isempty(nu) && throw(ArgumentError("frequency grid cannot be empty"))
    length(nu) == length(bp) ||
        throw(DimensionMismatch("frequency and passband vectors must have the same length"))
    all(isfinite, nu) || throw(ArgumentError("frequency grid must be finite"))
    all(isfinite, bp) || throw(ArgumentError("passband values must be finite"))
    @inbounds for index in 2:length(nu)
        nu[index] > nu[index - 1] ||
            throw(ArgumentError("frequency grid must be strictly increasing"))
    end
    return nothing
end

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

    function RawBand{T}(nu, bp) where T<:Real
        frequencies = convert(Vector{T}, nu)
        transmission = convert(Vector{T}, bp)
        _validate_band_inputs(frequencies, transmission)
        return new{T}(frequencies, transmission)
    end
end

RawBand(nu::Vector{T}, bp::Vector{T}) where T<:Real = RawBand{T}(nu, bp)

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

with `cmb2bb(ν) ∝ ∂B_ν/∂T`. A length-1 input with nonzero transmission
produces the same normalized monochromatic response as `point_band`.
"""
function make_band(nu::AbstractVector{T}, bp::AbstractVector{T}) where T<:Real
    _validate_band_inputs(nu, bp)
    if length(nu) == 1
        iszero(bp[1]) &&
            throw(DomainError(bp[1], "monochromatic transmission must be nonzero"))
        return Band{T}(Vector{T}(nu), T[one(T)], nu[1], true)
    end
    w       = bp .* cmb2bb.(nu)
    norm    = trapz(nu, w)
    isfinite(norm) && !iszero(norm) ||
        throw(DomainError(norm, "passband normalization must be finite and nonzero"))
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
        νmono::T = band.nu_eff
        return sed_fn(νmono)
    end

    return trapz(band.nu, sed_fn.(band.nu) .* band.norm_bp)
end

"""
    integrate_tsz(band, nu_0)

Integrate the normalized tSZ SED over a band.
This avoids higher-order closures in hot AD/JET paths.
"""
function integrate_tsz(band::Band{T}, nu_0::S, T_CMB::Real=T_CMB) where {T<:Real,S<:Real}
    if band.monofreq
        return tsz_sed(band.nu_eff, nu_0, T_CMB)
    end
    y = tsz_sed(band.nu, nu_0, T_CMB) .* band.norm_bp
    return trapz(band.nu, y)
end

@inline integrate_sed(sed_fn, band::DeltaBand) = sed_fn(band.nu_eff)
@inline integrate_tsz(band::DeltaBand, nu_0::Real, T_CMB::Real=T_CMB) =
    tsz_sed(band.nu_eff, nu_0, T_CMB)

"""
    eval_sed_bands(sed_fn, bands)

Evaluate a SED over an array of bands, returning one integrated SED value per
band. `sed_fn` must map a scalar frequency to a scalar response.
"""
function eval_sed_bands(sed_fn, bands::AbstractVector{<:AbstractBand})
    isempty(bands) && throw(ArgumentError("band collection cannot be empty"))
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
    PreparedChromaticBandpass

Reusable quadrature weights and chromatic normalization for one band and beam.
Construct with [`prepare_chromatic_bandpass`](@ref). Preparing outside repeated
SED evaluations treats the band and beam as fixed; preparing inside a
differentiated function preserves derivatives with respect to them.
For monochromatic bands, the beam cancels and its sampled frequency axis is not
used. Do not mutate the underlying band or beam arrays after preparation.
"""
struct PreparedChromaticBandpass{B<:AbstractBand, C<:ChromaticBeam, W, D}
    band::B
    beam::C
    weights::W
    denominator::D
end

@inline function _same_multipole_grid(a::AbstractVector, b::AbstractVector)
    length(a) == length(b) || return false
    @inbounds for i in eachindex(a, b)
        a[i] == b[i] || return false
    end
    return true
end


"""
    prepare_chromatic_bandpass(band, beam)

Prepare the SED-independent quadrature weights and normalized beam response for
reuse across foreground components.
"""
function prepare_chromatic_bandpass(band::Band, beam::ChromaticBeam)
    if band.monofreq
        # The beam cancels for a delta response, so its frequency axis is unused.
        return PreparedChromaticBandpass(band, beam, nothing, nothing)
    end

    size(beam.beam, 2) == length(band.nu) ||
        throw(DimensionMismatch("beam frequency dimension must match the band"))
    dnu = diff(band.nu)
    trapz_weights = vcat(first(dnu) / 2,
                         (dnu[1:end-1] .+ dnu[2:end]) ./ 2,
                         last(dnu) / 2)
    weights = trapz_weights .* band.norm_bp
    denominator = beam.beam * weights
    all(isfinite, denominator) && all(!iszero, denominator) ||
        throw(DomainError(denominator, "chromatic normalization must be finite and nonzero"))
    return PreparedChromaticBandpass(band, beam, weights, denominator)
end

@inline _fixed_beam_product(beam::AbstractMatrix, values::AbstractVector) = beam * values

"""
    prepare_fixed_chromatic_bandpass(band, beam) -> PreparedChromaticBandpass

Prepare a chromatic bandpass response while treating the beam as a **fixed**,
non-differentiable quantity.

This is the fixed-beam counterpart of [`prepare_chromatic_bandpass`](@ref). The
two produce numerically identical results; they differ only in reverse mode:

* `prepare_chromatic_bandpass` treats `beam.beam` as an active input and its
  pullback returns a dense cotangent of the same size as the beam matrix.
* `prepare_fixed_chromatic_bandpass` declares the beam constant, so its pullback
  returns `NoTangent()` for it.

Use this variant whenever the beam is a measured instrument calibration product
rather than an inference parameter. For a survey such as ACT DR6, whose
chromatic beams are `(n_ell, n_nu) = (8500, ~600)` per array, the fixed route
avoids allocating and accumulating one such matrix per SED per channel in every
reverse pass.

Derivatives with respect to the *band* are fully preserved, so a bandpass shift
applied with [`shift_and_normalize`](@ref) inside the differentiated call still
propagates correctly.

Pair it with [`eval_fixed_chromatic_sed_bands`](@ref).

```julia
bands    = [shift_and_normalize(raw, shift) for (raw, shift) in zip(raws, shifts)]
prepared = [prepare_fixed_chromatic_bandpass(band, beam)
            for (band, beam) in zip(bands, beams)]
weights  = eval_fixed_chromatic_sed_bands(nu -> sed_weight(sed, nu, beta), prepared)
```

For a monochromatic band the beam cancels identically and the result matches
`prepare_chromatic_bandpass` exactly. Do not mutate the underlying band or beam
arrays after preparation.
"""
function prepare_fixed_chromatic_bandpass(band::Band, beam::ChromaticBeam)
    if band.monofreq
        return PreparedChromaticBandpass(band, beam, nothing, nothing)
    end
    size(beam.beam, 2) == length(band.nu) ||
        throw(DimensionMismatch("beam frequency dimension must match the band"))
    dnu = diff(band.nu)
    trapz_weights = vcat(first(dnu) / 2,
                         (dnu[1:end-1] .+ dnu[2:end]) ./ 2,
                         last(dnu) / 2)
    weights = trapz_weights .* band.norm_bp
    denominator = _fixed_beam_product(beam.beam, weights)
    all(isfinite, denominator) && all(!iszero, denominator) ||
        throw(DomainError(denominator, "chromatic normalization must be finite and nonzero"))
    return PreparedChromaticBandpass(band, beam, weights, denominator)
end

"""
    prepare_fixed_chromatic_bandpass(band::DeltaBand, beam)

A `DeltaBand` has no sampled frequency grid and the beam cancels identically, so
the fixed and active routes coincide.
"""
prepare_fixed_chromatic_bandpass(band::DeltaBand, beam::ChromaticBeam) =
    PreparedChromaticBandpass(band, beam, nothing, nothing)

function prepare_chromatic_bandpass(band::DeltaBand, beam::ChromaticBeam)
    # A DeltaBand has no sampled frequency grid; the beam cancels identically.
    return PreparedChromaticBandpass(band, beam, nothing, nothing)
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
    prepared = prepare_chromatic_bandpass(band, chromatic_beam)
    return integrate_chromatic_sed(sed_fn, prepared)
end

function _chromatic_ratio(beam::AbstractMatrix, weights::AbstractVector,
                          sed_values::AbstractVector, denominator::AbstractVector)
    return (beam * (weights .* sed_values)) ./ denominator
end

function _fixed_chromatic_ratio(beam::AbstractMatrix, weights::AbstractVector,
                                sed_values::AbstractVector, denominator::AbstractVector)
    return (beam * (weights .* sed_values)) ./ denominator
end

@inline function integrate_chromatic_sed(sed_fn, prepared::PreparedChromaticBandpass)
    band = prepared.band
    if prepared.weights === nothing
        return fill(sed_fn(band.nu_eff), length(prepared.beam.ells))
    end
    return _chromatic_ratio(prepared.beam.beam, prepared.weights,
                            sed_fn.(band.nu), prepared.denominator)
end

@inline function _integrate_fixed_chromatic_sed(sed_fn,
                                                 prepared::PreparedChromaticBandpass)
    band = prepared.band
    if prepared.weights === nothing
        return fill(sed_fn(band.nu_eff), length(prepared.beam.ells))
    end
    return _fixed_chromatic_ratio(prepared.beam.beam, prepared.weights,
                                  sed_fn.(band.nu), prepared.denominator)
end

@inline integrate_chromatic_sed(sed_fn, band::DeltaBand,
                                chromatic_beam::ChromaticBeam) =
    integrate_chromatic_sed(sed_fn,
                            prepare_chromatic_bandpass(band, chromatic_beam))

"""
    eval_chromatic_sed_bands(sed_fn, bands, chromatic_beams) -> Matrix

Evaluate chromatic SED weights across an array of `Band`s and their corresponding `ChromaticBeam`s.
Returns a 2D matrix of shape `(n_freq, n_ell)`.
"""
function eval_chromatic_sed_bands(sed_fn, bands::AbstractVector{<:AbstractBand}, chromatic_beams::AbstractVector{<:ChromaticBeam})
    prepared = _prepare_chromatic_bandpasses(bands, chromatic_beams)
    return eval_chromatic_sed_bands(sed_fn, prepared)
end

function _prepare_chromatic_bandpasses(
    bands::AbstractVector{<:AbstractBand},
    chromatic_beams::AbstractVector{<:ChromaticBeam}
)
    Base.require_one_based_indexing(bands, chromatic_beams)
    n_freq = length(bands)
    n_freq > 0 || throw(ArgumentError("eval_chromatic_sed_bands: bands cannot be empty"))
    length(chromatic_beams) == n_freq ||
        throw(DimensionMismatch("number of bands and chromatic beams must match"))
    return [prepare_chromatic_bandpass(bands[i], chromatic_beams[i])
            for i in eachindex(bands)]
end

function eval_chromatic_sed_bands(
    sed_fn, prepared::AbstractVector{<:PreparedChromaticBandpass}
)
    Base.require_one_based_indexing(prepared)
    isempty(prepared) &&
        throw(ArgumentError("eval_chromatic_sed_bands: responses cannot be empty"))
    reference_ells = prepared[1].beam.ells
    all(response -> _same_multipole_grid(response.beam.ells, reference_ells), prepared) ||
        throw(ArgumentError("all prepared responses must use the same multipole grid"))
    responses = [integrate_chromatic_sed(sed_fn, response) for response in prepared]
    return permutedims(reduce(hcat, responses))
end

"""
    eval_fixed_chromatic_sed_bands(sed_fn, prepared) -> Matrix

Evaluate a SED across a vector of fixed-beam chromatic responses, returning an
`(n_freq, n_ell)` matrix.

This is the fixed-beam counterpart of [`eval_chromatic_sed_bands`](@ref) and
consumes the output of [`prepare_fixed_chromatic_bandpass`](@ref). The primal
result is identical to the active-beam route; only the reverse-mode behaviour
differs, in that the beam receives `NoTangent()`.

All prepared responses must share one multipole grid.
"""
function eval_fixed_chromatic_sed_bands(
    sed_fn, prepared::AbstractVector{<:PreparedChromaticBandpass}
)
    Base.require_one_based_indexing(prepared)
    isempty(prepared) &&
        throw(ArgumentError("fixed chromatic responses cannot be empty"))
    reference_ells = prepared[1].beam.ells
    all(response -> _same_multipole_grid(response.beam.ells, reference_ells), prepared) ||
        throw(ArgumentError("all prepared responses must use the same multipole grid"))
    responses = [_integrate_fixed_chromatic_sed(sed_fn, response) for response in prepared]
    return permutedims(reduce(hcat, responses))
end
