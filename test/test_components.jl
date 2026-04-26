"""
Component-registry tests (Step 5).

Each `compute_dl(component, ctx, p)` is checked against the equivalent
inline expression — the exact arithmetic that ACT's
`compute_fg_totals` performs today. The match is required at machine
precision (`isequal`), not just `≈`, because the registry is supposed
to be a refactoring of the current code, not an approximation of it.

Coverage:
  - KSZ                (TT)
  - TSZ                (TT)
  - CIBClustered       (TT)
  - CorrelatedTSZxCIB  (TT)              ← ACT's tSZ + CIBC + cross block
  - CIBPoisson         (TT)
  - Radio              (TT, TE, EE)
  - DustPL             (TT, TE, EE)
  - compute_fg_total   ACT TT == assemble_TT (rtol=1e-12)
                       ACT EE == assemble_EE (rtol=1e-12)
                       ACT TE == assemble_TE (rtol=1e-12)

Plus AD agreement (ForwardDiff vs Mooncake on a small parameter
vector) at rtol=1e-10 for each of the three spectra.
"""

using ADTypes
import DifferentiationInterface as DI
using ForwardDiff
using Mooncake
using Random


# ----------------------------------------------------------------- #
# Synthetic context — small enough for fast tests, real enough to    #
# exercise every code path.                                          #
# ----------------------------------------------------------------- #

function _make_test_context(::Val{S}; n_exp=3, n_ell=12, rng=MersenneTwister(0xACDC)) where S
    nu_0  = 150.0
    # Use a small ell_0 so that synthetic templates (sized ell_0+2) stay tiny.
    # eval_template reads T[ell_0+1], so the template must have at least ell_0+2 elements.
    ell   = collect(2:n_ell+1)                  # 2..13 for n_ell=12
    ell_0 = maximum(ell) + 2                    # = 15, safely inside template
    L     = ell_0 + 2                           # = 17; T[ell_0+1] = T[16] is in bounds

    # Gaussian bandpasses centered at 90 / 150 / 220 GHz
    centers_T = (90.0, 150.0, 220.0)
    centers_P = (95.0, 148.0, 225.0)
    function gaussband(ν0)
        νs = collect(range(ν0 - 30.0, ν0 + 30.0; length=21))
        bp = exp.(-((νs .- ν0) ./ 12.0).^2 ./ 2)
        return shift_and_normalize(RawBand{Float64}(νs, bp), 0.0)
    end
    bands_T = [gaussband(centers_T[i]) for i in 1:n_exp]
    bands_P = [gaussband(centers_P[i]) for i in 1:n_exp]

    # Random ℓ-templates of length L (indexed as T[ell+1], T[ell_0+1])
    templates = (
        T_tsz    = abs.(randn(rng, L)) .+ 0.1,
        T_ksz    = abs.(randn(rng, L)) .+ 0.1,
        T_cibc   = abs.(randn(rng, L)) .+ 0.1,
        T_szxcib = abs.(randn(rng, L)) .+ 0.1,
    )

    return FGContext(Val(S), ell, ell_0, nu_0, bands_T, bands_P, templates)
end

# Default ACT-style parameter NamedTuple
function _make_test_params(rng=MersenneTwister(0xBEEF))
    return (;
        a_kSZ      = 1.5 + 0.1 * rand(rng),
        a_tSZ      = 3.4 + 0.1 * rand(rng),
        alpha_tSZ  = -0.5,
        a_p        = 7.0,
        beta_p     = 2.05,
        a_c        = 4.7,
        xi         = 0.05,
        a_s        = 2.7,
        beta_s     = -2.5,
        a_gtt      = 8.0,
        a_pste     = 0.05,
        a_gte      = 0.42,
        a_psee     = 0.10,
        a_gee      = 0.17,
    )
end


# ----------------------------------------------------------------- #
# Per-component bit-for-bit equivalence                               #
# ----------------------------------------------------------------- #

@testset "KSZ — TT" begin
    ctx = _make_test_context(Val(:TT))
    p   = _make_test_params()

    # Inline reference (mirrors ACT's compute_fg_totals)
    f_ksz  = eval_sed_bands((ν::Float64) -> constant_sed(ν), ctx.bands_T)
    cl_ksz = ksz_template_scaled(eval_template(ctx.templates.T_ksz, ctx.ell, ctx.ell_0),
                                 p.a_kSZ)
    expected = factorized_cross(f_ksz, cl_ksz)

    @test compute_dl(KSZ(), ctx, p) == expected
end

@testset "TSZ — TT (standalone)" begin
    ctx = _make_test_context(Val(:TT))
    p   = _make_test_params()
    f_tsz  = eval_sed_bands((ν::Float64) -> tsz_sed(ν, ctx.nu_0), ctx.bands_T)
    cl_tsz = eval_template_tilt(ctx.templates.T_tsz, ctx.ell, ctx.ell_0,
                                p.alpha_tSZ; amp=p.a_tSZ)
    expected = factorized_cross(f_tsz, cl_tsz)
    @test compute_dl(TSZ(), ctx, p) == expected
end

@testset "CIBClustered — TT (standalone)" begin
    ctx = _make_test_context(Val(:TT))
    p   = _make_test_params()
    T_d    = 9.6
    beta_c = p.beta_p
    f_cibc = eval_sed_bands((ν::Float64) -> mbb_sed(ν, ctx.nu_0, beta_c, T_d),
                            ctx.bands_T)
    cl_cibc = eval_template(ctx.templates.T_cibc, ctx.ell, ctx.ell_0; amp=p.a_c)
    expected = factorized_cross(f_cibc, cl_cibc)
    @test compute_dl(CIBClustered(), ctx, p) == expected
end

@testset "CorrelatedTSZxCIB — TT (ACT 2×2 block)" begin
    ctx = _make_test_context(Val(:TT))
    p   = _make_test_params()
    T_d    = 9.6
    beta_c = p.beta_p
    f_tsz  = eval_sed_bands((ν::Float64) -> tsz_sed(ν, ctx.nu_0), ctx.bands_T)
    f_cibc = eval_sed_bands((ν::Float64) -> mbb_sed(ν, ctx.nu_0, beta_c, T_d),
                            ctx.bands_T)
    cl_tsz = eval_template_tilt(ctx.templates.T_tsz, ctx.ell, ctx.ell_0,
                                p.alpha_tSZ; amp=p.a_tSZ)
    cl_cibc   = eval_template(ctx.templates.T_cibc, ctx.ell, ctx.ell_0; amp=p.a_c)
    cl_szxcib = eval_template(ctx.templates.T_szxcib, ctx.ell, ctx.ell_0;
                              amp=-p.xi * sqrt(p.a_tSZ * p.a_c))
    F = vcat(reshape(f_tsz, 1, :), reshape(f_cibc, 1, :))
    C = build_szxcib_cl(cl_tsz, cl_cibc, cl_szxcib)
    expected = correlated_cross(F, C)
    @test compute_dl(CorrelatedTSZxCIB(), ctx, p) == expected
end

@testset "CIBPoisson — TT" begin
    ctx = _make_test_context(Val(:TT))
    p   = _make_test_params()
    T_d     = 9.6
    alpha_p = 1.0
    f_cibp  = eval_sed_bands((ν::Float64) -> mbb_sed(ν, ctx.nu_0, p.beta_p, T_d),
                             ctx.bands_T)
    ell_clp  = Float64.(ctx.ell .* (ctx.ell .+ 1))
    ell_0clp = Float64(ctx.ell_0 * (ctx.ell_0 + 1))
    cl_cibp = eval_powerlaw(ell_clp, ell_0clp, alpha_p)
    expected = p.a_p .* factorized_cross(f_cibp, cl_cibp)
    @test compute_dl(CIBPoisson(), ctx, p) == expected
end

@testset "Radio — TT/TE/EE" begin
    p = _make_test_params()
    ctx_tt = _make_test_context(Val(:TT))
    ctx_te = _make_test_context(Val(:TE))
    ctx_ee = _make_test_context(Val(:EE))

    f_radio_T = eval_sed_bands((ν::Float64) -> radio_sed(ν, ctx_tt.nu_0, p.beta_s),
                               ctx_tt.bands_T)
    f_radio_P = eval_sed_bands((ν::Float64) -> radio_sed(ν, ctx_tt.nu_0, p.beta_s),
                               ctx_tt.bands_P)
    ell_clp  = Float64.(ctx_tt.ell .* (ctx_tt.ell .+ 1))
    ell_0clp = Float64(ctx_tt.ell_0 * (ctx_tt.ell_0 + 1))
    cl_radio = eval_powerlaw(ell_clp, ell_0clp, 1.0)

    @test compute_dl(Radio(), ctx_tt, p) == p.a_s    .* factorized_cross(f_radio_T, cl_radio)
    @test compute_dl(Radio(), ctx_te, p) == p.a_pste .* factorized_cross_te(f_radio_T, f_radio_P, cl_radio)
    @test compute_dl(Radio(), ctx_ee, p) == p.a_psee .* factorized_cross(f_radio_P, cl_radio)
end

@testset "DustPL — TT/TE/EE" begin
    p = _make_test_params()
    ctx_tt = _make_test_context(Val(:TT))
    ctx_te = _make_test_context(Val(:TE))
    ctx_ee = _make_test_context(Val(:EE))

    f_dust_T = eval_sed_bands((ν::Float64) -> mbb_sed(ν, ctx_tt.nu_0, 1.5, 19.6),
                              ctx_tt.bands_T)
    f_dust_P = eval_sed_bands((ν::Float64) -> mbb_sed(ν, ctx_tt.nu_0, 1.5, 19.6),
                              ctx_tt.bands_P)
    cl_dustT = eval_powerlaw(Float64.(ctx_tt.ell), 500.0, -0.6)
    cl_dustE = eval_powerlaw(Float64.(ctx_tt.ell), 500.0, -0.4)

    @test compute_dl(DustPL(), ctx_tt, p) == p.a_gtt .* factorized_cross(f_dust_T, cl_dustT)
    @test compute_dl(DustPL(), ctx_te, p) == p.a_gte .* factorized_cross_te(f_dust_T, f_dust_P, cl_dustE)
    @test compute_dl(DustPL(), ctx_ee, p) == p.a_gee .* factorized_cross(f_dust_P, cl_dustE)
end


# ----------------------------------------------------------------- #
# Registry total == fused assemblers (the proof of refactoring)       #
# ----------------------------------------------------------------- #

@testset "compute_fg_total ACT TT == assemble_TT" begin
    ctx = _make_test_context(Val(:TT))
    p   = _make_test_params()

    components = (KSZ(), CorrelatedTSZxCIB(), CIBPoisson(), Radio(), DustPL())
    registry_TT = compute_fg_total(components, ctx, p)

    # Reference: build the inputs assemble_TT expects, then call it.
    f_ksz   = eval_sed_bands((ν::Float64) -> constant_sed(ν),                 ctx.bands_T)
    f_tsz   = eval_sed_bands((ν::Float64) -> tsz_sed(ν, ctx.nu_0),            ctx.bands_T)
    f_cibc  = eval_sed_bands((ν::Float64) -> mbb_sed(ν, ctx.nu_0, p.beta_p, 9.6), ctx.bands_T)
    f_cibp  = eval_sed_bands((ν::Float64) -> mbb_sed(ν, ctx.nu_0, p.beta_p, 9.6), ctx.bands_T)
    f_radio = eval_sed_bands((ν::Float64) -> radio_sed(ν, ctx.nu_0, p.beta_s),    ctx.bands_T)
    f_dust  = eval_sed_bands((ν::Float64) -> mbb_sed(ν, ctx.nu_0, 1.5, 19.6),     ctx.bands_T)

    cl_ksz   = ksz_template_scaled(eval_template(ctx.templates.T_ksz, ctx.ell, ctx.ell_0),
                                   p.a_kSZ)
    cl_tsz   = eval_template_tilt(ctx.templates.T_tsz, ctx.ell, ctx.ell_0,
                                  p.alpha_tSZ; amp=p.a_tSZ)
    cl_cibc  = eval_template(ctx.templates.T_cibc,   ctx.ell, ctx.ell_0; amp=p.a_c)
    cl_szxcib = eval_template(ctx.templates.T_szxcib, ctx.ell, ctx.ell_0;
                              amp=-p.xi * sqrt(p.a_tSZ * p.a_c))
    ell_clp  = Float64.(ctx.ell .* (ctx.ell .+ 1))
    ell_0clp = Float64(ctx.ell_0 * (ctx.ell_0 + 1))
    cl_cibp  = eval_powerlaw(ell_clp, ell_0clp, 1.0)
    cl_radio = eval_powerlaw(ell_clp, ell_0clp, 1.0)
    cl_dustT = eval_powerlaw(Float64.(ctx.ell), 500.0, -0.6)

    fused_TT = assemble_TT(p.a_p, p.a_gtt, p.a_s,
                           f_ksz, f_cibp, f_dust, f_radio, f_tsz, f_cibc,
                           cl_ksz, cl_cibp, cl_dustT, cl_radio,
                           cl_tsz, cl_cibc, cl_szxcib)

    @test registry_TT ≈ fused_TT rtol=1e-12
end

@testset "compute_fg_total ACT EE == assemble_EE" begin
    ctx = _make_test_context(Val(:EE))
    p   = _make_test_params()

    components = (Radio(), DustPL())
    registry_EE = compute_fg_total(components, ctx, p)

    f_radio_P = eval_sed_bands((ν::Float64) -> radio_sed(ν, ctx.nu_0, p.beta_s),
                               ctx.bands_P)
    f_dust_P  = eval_sed_bands((ν::Float64) -> mbb_sed(ν, ctx.nu_0, 1.5, 19.6),
                               ctx.bands_P)
    ell_clp  = Float64.(ctx.ell .* (ctx.ell .+ 1))
    ell_0clp = Float64(ctx.ell_0 * (ctx.ell_0 + 1))
    cl_radio = eval_powerlaw(ell_clp, ell_0clp, 1.0)
    cl_dustE = eval_powerlaw(Float64.(ctx.ell), 500.0, -0.4)

    fused_EE = assemble_EE(p.a_psee, p.a_gee, f_radio_P, f_dust_P, cl_radio, cl_dustE)
    @test registry_EE ≈ fused_EE rtol=1e-12
end

@testset "compute_fg_total ACT TE == assemble_TE" begin
    ctx = _make_test_context(Val(:TE))
    p   = _make_test_params()

    components = (Radio(), DustPL())
    registry_TE = compute_fg_total(components, ctx, p)

    f_radio_T = eval_sed_bands((ν::Float64) -> radio_sed(ν, ctx.nu_0, p.beta_s),
                               ctx.bands_T)
    f_radio_P = eval_sed_bands((ν::Float64) -> radio_sed(ν, ctx.nu_0, p.beta_s),
                               ctx.bands_P)
    f_dust_T  = eval_sed_bands((ν::Float64) -> mbb_sed(ν, ctx.nu_0, 1.5, 19.6),
                               ctx.bands_T)
    f_dust_P  = eval_sed_bands((ν::Float64) -> mbb_sed(ν, ctx.nu_0, 1.5, 19.6),
                               ctx.bands_P)
    ell_clp  = Float64.(ctx.ell .* (ctx.ell .+ 1))
    ell_0clp = Float64(ctx.ell_0 * (ctx.ell_0 + 1))
    cl_radio = eval_powerlaw(ell_clp, ell_0clp, 1.0)
    cl_dustE = eval_powerlaw(Float64.(ctx.ell), 500.0, -0.4)

    fused_TE = assemble_TE(p.a_pste, p.a_gte,
                           f_radio_T, f_radio_P, f_dust_T, f_dust_P,
                           cl_radio, cl_dustE)
    @test registry_TE ≈ fused_TE rtol=1e-12
end


# ----------------------------------------------------------------- #
# AD agreement                                                        #
# ----------------------------------------------------------------- #

@testset "compute_fg_total — ForwardDiff ≈ Mooncake (TT)" begin
    ctx = _make_test_context(Val(:TT))
    components = (KSZ(), CorrelatedTSZxCIB(), CIBPoisson(), Radio(), DustPL())

    function loss(v)
        p = (a_kSZ=v[1], a_tSZ=v[2], alpha_tSZ=v[3],
             a_p=v[4],   beta_p=v[5], a_c=v[6], xi=v[7],
             a_s=v[8],   beta_s=v[9], a_gtt=v[10])
        return sum(compute_fg_total(components, ctx, p))
    end
    v0 = [1.5, 3.4, -0.5, 7.0, 2.05, 4.7, 0.05, 2.7, -2.5, 8.0]

    grad_fd = DI.gradient(loss, AutoForwardDiff(), v0)
    grad_mk = DI.gradient(loss, AutoMooncake(; config=nothing), v0)
    @test grad_fd ≈ grad_mk rtol=1e-10
end

@testset "compute_fg_total — ForwardDiff ≈ Mooncake (TE)" begin
    ctx = _make_test_context(Val(:TE))
    components = (Radio(), DustPL())

    function loss(v)
        p = (a_pste=v[1], beta_s=v[2], a_gte=v[3])
        return sum(compute_fg_total(components, ctx, p))
    end
    v0 = [0.05, -2.5, 0.42]

    grad_fd = DI.gradient(loss, AutoForwardDiff(), v0)
    grad_mk = DI.gradient(loss, AutoMooncake(; config=nothing), v0)
    @test grad_fd ≈ grad_mk rtol=1e-10
end

@testset "compute_fg_total — ForwardDiff ≈ Mooncake (EE)" begin
    ctx = _make_test_context(Val(:EE))
    components = (Radio(), DustPL())

    function loss(v)
        p = (a_psee=v[1], beta_s=v[2], a_gee=v[3])
        return sum(compute_fg_total(components, ctx, p))
    end
    v0 = [0.10, -2.5, 0.17]

    grad_fd = DI.gradient(loss, AutoForwardDiff(), v0)
    grad_mk = DI.gradient(loss, AutoMooncake(; config=nothing), v0)
    @test grad_fd ≈ grad_mk rtol=1e-10
end
