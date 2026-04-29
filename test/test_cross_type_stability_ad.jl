"""
Type-stability (JET) + autodiff (DifferentiationInterface) tests for the
cross-spectrum kernels and their fused assemblers:

  factorized_cross, factorized_cross_te, correlated_cross,
  build_szxcib_cl, assemble_TT, assemble_EE, assemble_TE

Each function is checked for:
  1. Forward correctness — fused assemblers reproduce the naive
     element-wise composition exactly (rtol = 1e-12).
  2. Type stability — `JET.@test_opt` on a representative call.
  3. AD agreement — ForwardDiff vs Mooncake gradients of a scalar
     reduction must agree (rtol = 1e-10).

The Mooncake path goes through the `@from_chainrules` registrations in
`CMBForegroundsMooncakeExt`, so this also tests that the extension is
correctly wired to the rrules in `src/rrules.jl`.

Zygote is intentionally excluded — the fused assemblers use
`Array{T}(undef, n_freq, n_freq, n_ell)` + scalar in-place writes for
performance, which Zygote cannot differentiate. This matches the design
in ACT/SPT/Hillipop where only ForwardDiff and Mooncake are used.
"""

using JET
using ADTypes
import DifferentiationInterface as DI
using ForwardDiff
using Mooncake
using Random


# ----------------------------------------------------------------- #
# Test fixtures — small problem dimensions for fast tests             #
# ----------------------------------------------------------------- #

const _N_FREQ = 4
const _N_ELL  = 8

function _make_inputs(rng=MersenneTwister(0xC1055))
    n_freq, n_ell = _N_FREQ, _N_ELL
    f      = rand(rng, n_freq)
    fT     = rand(rng, n_freq)
    fE     = rand(rng, n_freq)
    cl     = rand(rng, n_ell)
    cl2    = rand(rng, n_ell)
    f_mat  = rand(rng, 2, n_freq)
    cl_3d  = rand(rng, 2, 2, n_ell)
    return (; f, fT, fE, cl, cl2, f_mat, cl_3d)
end


# ----------------------------------------------------------------- #
# factorized_cross                                                     #
# ----------------------------------------------------------------- #
@testset "factorized_cross — forward + type stability + AD" begin
    inp = _make_inputs()
    f, cl = inp.f, inp.cl
    n_freq, n_ell = length(f), length(cl)

    D = factorized_cross(f, cl)
    @test size(D) == (n_freq, n_freq, n_ell)

    # Forward correctness: D[i,j,ℓ] = f[i] f[j] cl[ℓ]
    for i in 1:n_freq, j in 1:n_freq, ℓ in 1:n_ell
        @test D[i, j, ℓ] ≈ f[i] * f[j] * cl[ℓ]
    end

    JET.@test_opt factorized_cross(f, cl)

    # AD: gradient of sum(D) wrt f and wrt cl
    g_f(x)  = sum(factorized_cross(x, cl))
    g_cl(x) = sum(factorized_cross(f, x))

    grad_f_fd = DI.gradient(g_f,  AutoForwardDiff(), f)
    grad_f_mk = DI.gradient(g_f,  AutoMooncake(; config=nothing), f)
    @test grad_f_fd ≈ grad_f_mk rtol=1e-10

    grad_cl_fd = DI.gradient(g_cl, AutoForwardDiff(), cl)
    grad_cl_mk = DI.gradient(g_cl, AutoMooncake(; config=nothing), cl)
    @test grad_cl_fd ≈ grad_cl_mk rtol=1e-10
end


# ----------------------------------------------------------------- #
# factorized_cross_te                                                  #
# ----------------------------------------------------------------- #
@testset "factorized_cross_te — forward + type stability + AD" begin
    inp = _make_inputs()
    fT, fE, cl = inp.fT, inp.fE, inp.cl
    n_freq, n_ell = length(fT), length(cl)

    D = factorized_cross_te(fT, fE, cl)
    @test size(D) == (n_freq, n_freq, n_ell)

    for i in 1:n_freq, j in 1:n_freq, ℓ in 1:n_ell
        @test D[i, j, ℓ] ≈ fT[i] * fE[j] * cl[ℓ]
    end

    JET.@test_opt factorized_cross_te(fT, fE, cl)

    # Joint gradient wrt (fT, fE, cl) packed into one vector
    function g_pack(v)
        n = _N_FREQ
        fTv = v[1:n]
        fEv = v[n+1:2n]
        clv = v[2n+1:end]
        return sum(factorized_cross_te(fTv, fEv, clv))
    end
    v0 = vcat(fT, fE, cl)

    grad_fd = DI.gradient(g_pack, AutoForwardDiff(), v0)
    grad_mk = DI.gradient(g_pack, AutoMooncake(; config=nothing), v0)
    @test grad_fd ≈ grad_mk rtol=1e-10
end


# ----------------------------------------------------------------- #
# correlated_cross                                                     #
# ----------------------------------------------------------------- #
@testset "correlated_cross — forward + type stability + AD" begin
    inp = _make_inputs()
    f, cl = inp.f_mat, inp.cl_3d
    n_comp, n_freq = size(f)
    n_ell = size(cl, 3)

    D = correlated_cross(f, cl)
    @test size(D) == (n_freq, n_freq, n_ell)

    # Forward correctness: D[i,j,ℓ] = Σ_{k,n} f[k,i] f[n,j] C[k,n,ℓ]
    for i in 1:n_freq, j in 1:n_freq, ℓ in 1:n_ell
        ref = sum(f[k, i] * f[n, j] * cl[k, n, ℓ] for k in 1:n_comp, n in 1:n_comp)
        @test D[i, j, ℓ] ≈ ref
    end

    # JET 0.9.x (Julia 1.10) detects a spurious runtime dispatch inside
    # sum(generator over ProductIterator) in Base — not a real code issue.
    # The check is clean on JET ≥ 0.11 (Julia ≥ 1.11).
    if VERSION >= v"1.11"
        JET.@test_opt correlated_cross(f, cl)
    end

    # AD wrt f
    g_f(x) = sum(correlated_cross(x, cl))
    grad_fd = DI.gradient(g_f, AutoForwardDiff(), f)
    grad_mk = DI.gradient(g_f, AutoMooncake(; config=nothing), f)
    @test grad_fd ≈ grad_mk rtol=1e-10

    # AD wrt cl
    g_cl(x) = sum(correlated_cross(f, x))
    grad_fd = DI.gradient(g_cl, AutoForwardDiff(), cl)
    grad_mk = DI.gradient(g_cl, AutoMooncake(; config=nothing), cl)
    @test grad_fd ≈ grad_mk rtol=1e-10
end


# ----------------------------------------------------------------- #
# build_szxcib_cl                                                      #
# ----------------------------------------------------------------- #
@testset "build_szxcib_cl — forward + AD" begin
    rng = MersenneTwister(0xBADC0FFEE)
    n_ell  = _N_ELL
    cl_tsz   = rand(rng, n_ell)
    cl_cibc  = rand(rng, n_ell)
    cl_cross = rand(rng, n_ell)

    C = build_szxcib_cl(cl_tsz, cl_cibc, cl_cross)
    @test size(C) == (2, 2, n_ell)
    for ℓ in 1:n_ell
        @test C[1, 1, ℓ] ≈ cl_tsz[ℓ]
        @test C[2, 2, ℓ] ≈ cl_cibc[ℓ]
        @test C[1, 2, ℓ] ≈ cl_cross[ℓ]
        @test C[2, 1, ℓ] ≈ cl_cross[ℓ]
    end

    g(v) = sum(build_szxcib_cl(v[1:n_ell], v[n_ell+1:2n_ell], v[2n_ell+1:end]))
    v0 = vcat(cl_tsz, cl_cibc, cl_cross)
    grad_fd = DI.gradient(g, AutoForwardDiff(), v0)
    grad_mk = DI.gradient(g, AutoMooncake(; config=nothing), v0)
    @test grad_fd ≈ grad_mk rtol=1e-10
end


# ----------------------------------------------------------------- #
# assemble_TT — fused TT assembler                                     #
# ----------------------------------------------------------------- #
@testset "assemble_TT — forward equivalence + type stability + AD" begin
    rng = MersenneTwister(0x715071)
    n_freq, n_ell = _N_FREQ, _N_ELL

    a_p, a_gtt, a_s = 0.5, 0.7, 0.3
    f_ksz   = rand(rng, n_freq)
    f_cibp  = rand(rng, n_freq)
    f_dust  = rand(rng, n_freq)
    f_radio = rand(rng, n_freq)
    f_tsz   = rand(rng, n_freq)
    f_cibc  = rand(rng, n_freq)
    cl_ksz   = rand(rng, n_ell)
    cl_cibp  = rand(rng, n_ell)
    cl_dustT = rand(rng, n_ell)
    cl_radio = rand(rng, n_ell)
    cl_tsz   = rand(rng, n_ell)
    cl_cibc  = rand(rng, n_ell)
    cl_szxcib = rand(rng, n_ell)

    D = assemble_TT(a_p, a_gtt, a_s,
                    f_ksz, f_cibp, f_dust, f_radio, f_tsz, f_cibc,
                    cl_ksz, cl_cibp, cl_dustT, cl_radio,
                    cl_tsz, cl_cibc, cl_szxcib)

    # Reference via slow composition through factorized_cross + correlated_cross
    f_corr = vcat(f_tsz', f_cibc')   # (2, n_freq)
    cl_corr = build_szxcib_cl(cl_tsz, cl_cibc, cl_szxcib)
    D_ref =
        factorized_cross(f_ksz,   cl_ksz)             .+
        correlated_cross(f_corr,  cl_corr)            .+
        a_p   .* factorized_cross(f_cibp,  cl_cibp)  .+
        a_gtt .* factorized_cross(f_dust,  cl_dustT) .+
        a_s   .* factorized_cross(f_radio, cl_radio)
    @test D ≈ D_ref rtol=1e-12

    JET.@test_opt assemble_TT(a_p, a_gtt, a_s,
                              f_ksz, f_cibp, f_dust, f_radio, f_tsz, f_cibc,
                              cl_ksz, cl_cibp, cl_dustT, cl_radio,
                              cl_tsz, cl_cibc, cl_szxcib)

    # AD: gradient of sum(D) wrt the full parameter vector
    function g_TT(v)
        ap, ag, as = v[1], v[2], v[3]
        i = 4
        f_ksz   = v[i:i+n_freq-1]; i += n_freq
        f_cibp  = v[i:i+n_freq-1]; i += n_freq
        f_dust  = v[i:i+n_freq-1]; i += n_freq
        f_radio = v[i:i+n_freq-1]; i += n_freq
        f_tsz   = v[i:i+n_freq-1]; i += n_freq
        f_cibc  = v[i:i+n_freq-1]; i += n_freq
        cl_ksz   = v[i:i+n_ell-1]; i += n_ell
        cl_cibp  = v[i:i+n_ell-1]; i += n_ell
        cl_dustT = v[i:i+n_ell-1]; i += n_ell
        cl_radio = v[i:i+n_ell-1]; i += n_ell
        cl_tsz   = v[i:i+n_ell-1]; i += n_ell
        cl_cibc  = v[i:i+n_ell-1]; i += n_ell
        cl_szxcib = v[i:i+n_ell-1]
        return sum(assemble_TT(ap, ag, as,
                               f_ksz, f_cibp, f_dust, f_radio, f_tsz, f_cibc,
                               cl_ksz, cl_cibp, cl_dustT, cl_radio,
                               cl_tsz, cl_cibc, cl_szxcib))
    end
    v0 = vcat([a_p, a_gtt, a_s],
              f_ksz, f_cibp, f_dust, f_radio, f_tsz, f_cibc,
              cl_ksz, cl_cibp, cl_dustT, cl_radio,
              cl_tsz, cl_cibc, cl_szxcib)

    grad_fd = DI.gradient(g_TT, AutoForwardDiff(), v0)
    grad_mk = DI.gradient(g_TT, AutoMooncake(; config=nothing), v0)
    @test grad_fd ≈ grad_mk rtol=1e-10
end


# ----------------------------------------------------------------- #
# assemble_EE — fused EE assembler                                     #
# ----------------------------------------------------------------- #
@testset "assemble_EE — forward equivalence + type stability + AD" begin
    rng = MersenneTwister(0xEEEE)
    n_freq, n_ell = _N_FREQ, _N_ELL

    a_psee, a_gee = 0.4, 0.6
    f_radio_P = rand(rng, n_freq)
    f_dust_P  = rand(rng, n_freq)
    cl_radio  = rand(rng, n_ell)
    cl_dustE  = rand(rng, n_ell)

    D = assemble_EE(a_psee, a_gee, f_radio_P, f_dust_P, cl_radio, cl_dustE)
    D_ref =
        a_psee .* factorized_cross(f_radio_P, cl_radio) .+
        a_gee  .* factorized_cross(f_dust_P,  cl_dustE)
    @test D ≈ D_ref rtol=1e-12

    JET.@test_opt assemble_EE(a_psee, a_gee, f_radio_P, f_dust_P, cl_radio, cl_dustE)

    function g_EE(v)
        ap, ag = v[1], v[2]
        i = 3
        f_rP = v[i:i+n_freq-1]; i += n_freq
        f_dP = v[i:i+n_freq-1]; i += n_freq
        cl_r = v[i:i+n_ell-1];  i += n_ell
        cl_d = v[i:i+n_ell-1]
        return sum(assemble_EE(ap, ag, f_rP, f_dP, cl_r, cl_d))
    end
    v0 = vcat([a_psee, a_gee], f_radio_P, f_dust_P, cl_radio, cl_dustE)
    grad_fd = DI.gradient(g_EE, AutoForwardDiff(), v0)
    grad_mk = DI.gradient(g_EE, AutoMooncake(; config=nothing), v0)
    @test grad_fd ≈ grad_mk rtol=1e-10
end


# ----------------------------------------------------------------- #
# assemble_TE — fused TE assembler                                     #
# ----------------------------------------------------------------- #
@testset "assemble_TE — forward equivalence + type stability + AD" begin
    rng = MersenneTwister(0x7373)
    n_freq, n_ell = _N_FREQ, _N_ELL

    a_pste, a_gte = 0.2, 0.8
    f_radio_T = rand(rng, n_freq)
    f_radio_P = rand(rng, n_freq)
    f_dust_T  = rand(rng, n_freq)
    f_dust_P  = rand(rng, n_freq)
    cl_radio  = rand(rng, n_ell)
    cl_dustE  = rand(rng, n_ell)

    D = assemble_TE(a_pste, a_gte,
                    f_radio_T, f_radio_P, f_dust_T, f_dust_P,
                    cl_radio, cl_dustE)
    D_ref =
        a_pste .* factorized_cross_te(f_radio_T, f_radio_P, cl_radio) .+
        a_gte  .* factorized_cross_te(f_dust_T,  f_dust_P,  cl_dustE)
    @test D ≈ D_ref rtol=1e-12

    JET.@test_opt assemble_TE(a_pste, a_gte,
                              f_radio_T, f_radio_P, f_dust_T, f_dust_P,
                              cl_radio, cl_dustE)

    function g_TE(v)
        ap, ag = v[1], v[2]
        i = 3
        f_rT = v[i:i+n_freq-1]; i += n_freq
        f_rP = v[i:i+n_freq-1]; i += n_freq
        f_dT = v[i:i+n_freq-1]; i += n_freq
        f_dP = v[i:i+n_freq-1]; i += n_freq
        cl_r = v[i:i+n_ell-1];  i += n_ell
        cl_d = v[i:i+n_ell-1]
        return sum(assemble_TE(ap, ag, f_rT, f_rP, f_dT, f_dP, cl_r, cl_d))
    end
    v0 = vcat([a_pste, a_gte], f_radio_T, f_radio_P, f_dust_T, f_dust_P,
              cl_radio, cl_dustE)
    grad_fd = DI.gradient(g_TE, AutoForwardDiff(), v0)
    grad_mk = DI.gradient(g_TE, AutoMooncake(; config=nothing), v0)
    @test grad_fd ≈ grad_mk rtol=1e-10
end
