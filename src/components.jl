"""
    components.jl

Component-registry layer (Step 5 of the unification plan).

A foreground component is a singleton subtype of `FGComponent`. Each
component declares how it produces a per-spectrum 3-D `D_ℓ` array of
shape `(n_freq, n_freq, n_ell)` via `compute_dl(c, ctx, p)`.

The accumulator `compute_fg_total(components, ctx, p)` simply sums the
contributions, so survey-specific behavior is fully captured by the
component list and the parameter NamedTuple.

Each survey:
  • picks its component list (e.g. ACT TT = `(KSZ, CIBPoisson,
    CorrelatedTSZxCIB, Radio, DustPL)`);
  • supplies its parameter NamedTuple using ACT-style canonical names
    (`a_kSZ`, `a_tSZ`, `alpha_tSZ`, `a_p`, `beta_p`, `a_c`, `a_s`,
    `beta_s`, `a_gtt`, …) — adapter shims live in the survey
    package, not here.

The dispatch is fully type-stable: the spectrum (`:TT`, `:TE`, `:EE`)
is encoded as a `Val` type parameter on `FGContext`, so each
spectrum gets its own specialized method body.

Performance: each `compute_dl` call goes through one of the existing
`factorized_cross`/`factorized_cross_te`/`correlated_cross` rrules
(Step 4), so each component contributes one Mooncake tape entry. The
fused per-spectrum `assemble_TT/EE/TE` paths remain available as a
fast path; ACT chooses between them in `foreground.jl`.
"""

using ChainRulesCore: @ignore_derivatives

# ------------------------------------------------------------------ #
# Component types                                                      #
# ------------------------------------------------------------------ #

"""
    FGComponent

Abstract base type for all foreground components.
Concrete subtypes are singletons (no fields).
"""
abstract type FGComponent end

"Kinematic SZ — flat amplitude × kSZ template, frequency-flat (constant SED)."
struct KSZ <: FGComponent end

"Thermal SZ standalone — `factorized_cross(f_tsz, cl_tsz)`. Use when the
survey treats tSZ as an isolated component (SPT). For ACT/Hillipop where
tSZ is part of a 2×2 correlated block with CIBC, use `CorrelatedTSZxCIB`."
struct TSZ <: FGComponent end

"CIB Poisson — flat C_ℓ × MBB². Frequency: `mbb_sed(ν, ν₀, β_p, T_d)`."
struct CIBPoisson <: FGComponent end

"CIB clustered standalone — `factorized_cross(f_cibc, cl_cibc)`. Same
note as `TSZ` regarding the `CorrelatedTSZxCIB` block."
struct CIBClustered <: FGComponent end

"""
    CorrelatedTSZxCIB

The Addison+12 / fgspectra-style correlated 2×2 (tSZ, CIBC) block:

    Σ_{a,b ∈ {tSZ, CIBC}} f_a[i] f_b[j] C_{ab}(ℓ)

with `C_tSZ,tSZ = cl_tsz`, `C_CIBC,CIBC = cl_cibc`, and
`C_tSZ,CIBC = C_CIBC,tSZ = cl_szxcib = -ξ √(a_tSZ a_c) T_szxcib(ℓ)`.

Used by ACT DR6 and Hillipop. SPT uses `TSZxCIBAuto` instead.
"""
struct CorrelatedTSZxCIB <: FGComponent end

"Radio point sources — `factorized_cross(f_radio, cl_radio)` with
`f_radio = radio_sed(ν, ν₀, β_s)` and `cl_radio = (ℓ(ℓ+1)/ℓ₀(ℓ₀+1))^α_s`."
struct Radio <: FGComponent end

"Galactic dust, power-law in ℓ × MBB² in ν. Used by ACT and SPT."
struct DustPL <: FGComponent end

"tSZ × CIB auto-cross flavor — `−ξ × (√(tsz·cib) + √(tsz·cib))`. Used by SPT."
struct TSZxCIBAuto <: FGComponent end

"Galactic dust with per-cross-frequency template (Hillipop)."
struct DustTemplate <: FGComponent end

"Generic shot-noise term (SPT). Single amplitude per cross-frequency pair."
struct ShotNoise <: FGComponent end

"Sub-pixel HEALPix term (Hillipop)."
struct SubPixel <: FGComponent end


# ------------------------------------------------------------------ #
# FGContext                                                            #
# ------------------------------------------------------------------ #

"""
    FGContext{S,BT,NT}(ell, ell_0, nu_0, bands_T, bands_P, templates)

Spectrum-specialized context bundling everything a component needs
besides the parameter NamedTuple `p`:

  - `ell`        :: Vector{Int}      — multipole grid
  - `ell_0`      :: Int              — reference multipole (3000)
  - `nu_0`       :: Float64          — reference frequency (150 GHz);
                                       always a physical constant, never
                                       differentiated, so stays Float64
  - `bands_T`    :: Vector{Band{BT}} — bandpass-shifted T bands (1 per exp)
  - `bands_P`    :: Vector{Band{BT}} — bandpass-shifted P bands (1 per exp)
  - `templates`  :: NT               — `T_tsz`, `T_ksz`, `T_cibc`,
                                       `T_szxcib`, plus survey-specific
                                       templates (e.g. dust template
                                       arrays for Hillipop)

`BT` is the **band element type** — `Float64` at inference time, but
`Dual{...,Float64,...}` when ForwardDiff differentiates through
bandpass-shift parameters (which shift the frequency grid). Keeping
`BT` separate from `nu_0`'s type (always `Float64`) avoids a type
conflict: shifting bands under AD produces `Band{Dual}`, but `nu_0`
is a reference constant and never becomes Dual.

The spectrum (`:TT`, `:TE`, `:EE`) is the `S` type parameter so that
`compute_dl(c, ctx, p)` dispatches to the spectrum-correct method.
"""
struct FGContext{S, BT<:Real, NT<:NamedTuple}
    ell       :: Vector{Int}
    ell_0     :: Int
    nu_0      :: Float64
    bands_T   :: Vector{Band{BT}}
    bands_P   :: Vector{Band{BT}}
    templates :: NT
end

# Constructor sugar — picks up `S` from a `Val(:TT/:TE/:EE)`
function FGContext(spectrum::Val{S},
                   ell::AbstractVector{<:Integer},
                   ell_0::Integer,
                   nu_0::Real,
                   bands_T::Vector{Band{BT}},
                   bands_P::Vector{Band{BT}},
                   templates::NT) where {S, BT<:Real, NT<:NamedTuple}
    return FGContext{S,BT,NT}(Vector{Int}(ell), Int(ell_0), Float64(nu_0),
                              bands_T, bands_P, templates)
end


# ------------------------------------------------------------------ #
# Helpers — typed SED evaluator + ℓ-grid lookups                       #
# ------------------------------------------------------------------ #

# eval_sed_typed: forces the closure input type to `BT` (the band
# element type) so Julia infers a concrete return type.
@inline _eval_sed_typed(f::F, bands::AbstractVector{Band{BT}}) where {F, BT<:Real} =
    eval_sed_bands((ν::BT) -> f(ν), bands)

# ℓ(ℓ+1) grid for Poisson/radio. Returned as Float64 so eval_powerlaw
# stays in Float64 regardless of `p`'s eltype.
@inline _ell_clp(ctx::FGContext) = (Float64.(ctx.ell .* (ctx.ell .+ 1)),
                                    Float64(ctx.ell_0 * (ctx.ell_0 + 1)))


# ------------------------------------------------------------------ #
# Parameter access — same `fg_param` semantics as ACT                  #
# ------------------------------------------------------------------ #
# We re-export ACT-style canonical names; surveys map to these at the
# `compute_fg_total` entry point. This keeps the registry math
# survey-agnostic without an extra runtime alias-lookup layer.

@inline _fg_param(p::NamedTuple, key::Symbol, default) =
    hasproperty(p, key) ? getproperty(p, key) : default

@inline _fg_param(p::AbstractDict, key::Symbol, default) =
    get(p, key, get(p, String(key), default))


# ------------------------------------------------------------------ #
# compute_dl methods — one per (component, spectrum) pair               #
# ------------------------------------------------------------------ #

# ----- KSZ (TT only) -----
function compute_dl(::KSZ, ctx::FGContext{:TT}, p)
    f_ksz  = _eval_sed_typed(ν -> constant_sed(ν), ctx.bands_T)
    cl_ksz = ksz_template_scaled(eval_template(ctx.templates.T_ksz, ctx.ell, ctx.ell_0),
                                 p.a_kSZ)
    return factorized_cross(f_ksz, cl_ksz)
end

# ----- TSZ standalone (TT only; for SPT) -----
function compute_dl(::TSZ, ctx::FGContext{:TT}, p)
    f_tsz  = _eval_sed_typed(ν -> tsz_sed(ν, ctx.nu_0), ctx.bands_T)
    cl_tsz = eval_template_tilt(ctx.templates.T_tsz, ctx.ell, ctx.ell_0,
                                _fg_param(p, :alpha_tSZ, 0.0);
                                amp=p.a_tSZ)
    return factorized_cross(f_tsz, cl_tsz)
end

# ----- CIBClustered standalone (TT only; for SPT) -----
function compute_dl(::CIBClustered, ctx::FGContext{:TT}, p)
    T_d    = _fg_param(p, :T_d,   9.6)
    beta_c = _fg_param(p, :beta_c, p.beta_p)
    f_cibc = _eval_sed_typed(ν -> mbb_sed(ν, ctx.nu_0, beta_c, T_d), ctx.bands_T)
    cl_cibc = eval_template(ctx.templates.T_cibc, ctx.ell, ctx.ell_0; amp=p.a_c)
    return factorized_cross(f_cibc, cl_cibc)
end

# ----- CorrelatedTSZxCIB (TT only; ACT/Hillipop flavor) -----
function compute_dl(::CorrelatedTSZxCIB, ctx::FGContext{:TT}, p)
    T_d    = _fg_param(p, :T_d,    9.6)
    beta_c = _fg_param(p, :beta_c, p.beta_p)
    f_tsz  = _eval_sed_typed(ν -> tsz_sed(ν, ctx.nu_0), ctx.bands_T)
    f_cibc = _eval_sed_typed(ν -> mbb_sed(ν, ctx.nu_0, beta_c, T_d), ctx.bands_T)

    cl_tsz = eval_template_tilt(ctx.templates.T_tsz, ctx.ell, ctx.ell_0,
                                _fg_param(p, :alpha_tSZ, 0.0);
                                amp=p.a_tSZ)
    cl_cibc   = eval_template(ctx.templates.T_cibc,   ctx.ell, ctx.ell_0; amp=p.a_c)
    cl_szxcib = eval_template(ctx.templates.T_szxcib, ctx.ell, ctx.ell_0;
                              amp=-p.xi * sqrt(p.a_tSZ * p.a_c))

    F = vcat(reshape(f_tsz, 1, :), reshape(f_cibc, 1, :))   # (2, n_freq)
    C = build_szxcib_cl(cl_tsz, cl_cibc, cl_szxcib)
    return correlated_cross(F, C)
end

# ----- CIB Poisson (TT only) -----
function compute_dl(::CIBPoisson, ctx::FGContext{:TT}, p)
    T_d     = _fg_param(p, :T_d,     9.6)
    alpha_p = _fg_param(p, :alpha_p, 1.0)
    f_cibp  = _eval_sed_typed(ν -> mbb_sed(ν, ctx.nu_0, p.beta_p, T_d), ctx.bands_T)
    ell_clp, ell_0clp = _ell_clp(ctx)
    cl_cibp = eval_powerlaw(ell_clp, ell_0clp, alpha_p)
    return p.a_p .* factorized_cross(f_cibp, cl_cibp)
end

# ----- Radio (TT, TE, EE) -----
function compute_dl(::Radio, ctx::FGContext{:TT}, p)
    alpha_s   = _fg_param(p, :alpha_s, 1.0)
    f_radio_T = _eval_sed_typed(ν -> radio_sed(ν, ctx.nu_0, p.beta_s), ctx.bands_T)
    ell_clp, ell_0clp = _ell_clp(ctx)
    cl_radio  = eval_powerlaw(ell_clp, ell_0clp, alpha_s)
    return p.a_s .* factorized_cross(f_radio_T, cl_radio)
end

function compute_dl(::Radio, ctx::FGContext{:TE}, p)
    alpha_s   = _fg_param(p, :alpha_s, 1.0)
    f_radio_T = _eval_sed_typed(ν -> radio_sed(ν, ctx.nu_0, p.beta_s), ctx.bands_T)
    f_radio_P = _eval_sed_typed(ν -> radio_sed(ν, ctx.nu_0, p.beta_s), ctx.bands_P)
    ell_clp, ell_0clp = _ell_clp(ctx)
    cl_radio  = eval_powerlaw(ell_clp, ell_0clp, alpha_s)
    return p.a_pste .* factorized_cross_te(f_radio_T, f_radio_P, cl_radio)
end

function compute_dl(::Radio, ctx::FGContext{:EE}, p)
    alpha_s   = _fg_param(p, :alpha_s, 1.0)
    f_radio_P = _eval_sed_typed(ν -> radio_sed(ν, ctx.nu_0, p.beta_s), ctx.bands_P)
    ell_clp, ell_0clp = _ell_clp(ctx)
    cl_radio  = eval_powerlaw(ell_clp, ell_0clp, alpha_s)
    return p.a_psee .* factorized_cross(f_radio_P, cl_radio)
end

# ----- DustPL (TT, TE, EE) -----
function compute_dl(::DustPL, ctx::FGContext{:TT}, p)
    beta_d   = _fg_param(p, :beta_d,   1.5)
    T_effd   = _fg_param(p, :T_effd,   19.6)
    alpha_dT = _fg_param(p, :alpha_dT, -0.6)
    f_dust_T = _eval_sed_typed(ν -> mbb_sed(ν, ctx.nu_0, beta_d, T_effd), ctx.bands_T)
    cl_dustT = eval_powerlaw(Float64.(ctx.ell), 500.0, alpha_dT)
    return p.a_gtt .* factorized_cross(f_dust_T, cl_dustT)
end

function compute_dl(::DustPL, ctx::FGContext{:TE}, p)
    beta_d   = _fg_param(p, :beta_d,   1.5)
    T_effd   = _fg_param(p, :T_effd,   19.6)
    alpha_dE = _fg_param(p, :alpha_dE, -0.4)
    f_dust_T = _eval_sed_typed(ν -> mbb_sed(ν, ctx.nu_0, beta_d, T_effd), ctx.bands_T)
    f_dust_P = _eval_sed_typed(ν -> mbb_sed(ν, ctx.nu_0, beta_d, T_effd), ctx.bands_P)
    cl_dustE = eval_powerlaw(Float64.(ctx.ell), 500.0, alpha_dE)
    return p.a_gte .* factorized_cross_te(f_dust_T, f_dust_P, cl_dustE)
end

function compute_dl(::DustPL, ctx::FGContext{:EE}, p)
    beta_d   = _fg_param(p, :beta_d,   1.5)
    T_effd   = _fg_param(p, :T_effd,   19.6)
    alpha_dE = _fg_param(p, :alpha_dE, -0.4)
    f_dust_P = _eval_sed_typed(ν -> mbb_sed(ν, ctx.nu_0, beta_d, T_effd), ctx.bands_P)
    cl_dustE = eval_powerlaw(Float64.(ctx.ell), 500.0, alpha_dE)
    return p.a_gee .* factorized_cross(f_dust_P, cl_dustE)
end


# ------------------------------------------------------------------ #
# Accumulator                                                          #
# ------------------------------------------------------------------ #

"""
    compute_fg_total(components, ctx, p) → Array{T,3}

Sum of `compute_dl(c, ctx, p)` over all `c ∈ components`. The
component collection should be a `Tuple` so the sum is fully
type-inferred and unrolled.
"""
@inline function compute_fg_total(components::Tuple, ctx::FGContext, p)
    return _accumulate(components, ctx, p)
end

# Recursive accumulator — guarantees type stability when `components`
# is a heterogeneous Tuple.
@inline _accumulate(::Tuple{}, ctx, p) =
    error("compute_fg_total: empty component list")

@inline _accumulate(cs::Tuple{Any}, ctx, p) =
    compute_dl(cs[1], ctx, p)

@inline _accumulate(cs::Tuple, ctx, p) =
    compute_dl(cs[1], ctx, p) .+ _accumulate(Base.tail(cs), ctx, p)
