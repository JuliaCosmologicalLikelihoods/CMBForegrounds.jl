"""
Type-stability (JET) + autodiff (DifferentiationInterface) tests for
the two template-based foreground D_ℓ helpers used by Hillipop:

  cib_clustered_template_power(template, A_CIB, β, Tdust, ν0_cib, ν1, ν2)
  tsz_cib_template_power(template, ξ, A_tSZ, A_CIB, β, Tdust, ν0_tsz, ν0_cib,
                          ν_sz1, ν_sz2, ν_cib1, ν_cib2)

Free parameters we differentiate w.r.t. are physically-meaningful subset
(amplitudes, spectral index, ξ).  Frequencies are constants (effective
band centers).
"""

using JET
using ADTypes
import DifferentiationInterface as DI
using ForwardDiff
using Zygote
using Mooncake

@testset "cib_clustered_template_power — type stability + AD" begin

    tmpl    = collect(range(0.5, 1.5, length=100))
    Acib    = 4.0
    β       = 1.75
    Tdust   = 25.0
    ν0_cib  = 143.0
    ν1, ν2  = 147.5, 228.1

    w = randn(MersenneTwister(11), length(tmpl))

    # Loss over [Acib, β]
    f = x -> sum(w .* CMBForegrounds.cib_clustered_template_power(
        tmpl, x[1], x[2], Tdust, ν0_cib, ν1, ν2))
    x0 = [Acib, β]

    @testset "JET @test_opt — Float64" begin
        JET.@test_opt CMBForegrounds.cib_clustered_template_power(
            tmpl, Acib, β, Tdust, ν0_cib, ν1, ν2)
    end

    @testset "JET @test_opt — Dual β" begin
        β_d = ForwardDiff.Dual(β, 1.0)
        JET.@test_opt target_modules=(CMBForegrounds,) CMBForegrounds.cib_clustered_template_power(
            tmpl, Acib, β_d, Tdust, ν0_cib, ν1, ν2)
    end

    @testset "DI gradient — ForwardDiff" begin
        g = DI.gradient(f, AutoForwardDiff(), x0)
        @test length(g) == 2
        @test all(isfinite, g)
    end

    @testset "DI gradient — Zygote" begin
        g = DI.gradient(f, AutoZygote(), x0)
        @test all(isfinite, g)
    end

    @testset "DI gradient — Mooncake" begin
        g = DI.gradient(f, AutoMooncake(; config=nothing), x0)
        @test all(isfinite, g)
    end

    @testset "Cross-backend agreement" begin
        gFD = DI.gradient(f, AutoForwardDiff(),              x0)
        gZG = DI.gradient(f, AutoZygote(),                   x0)
        gMC = DI.gradient(f, AutoMooncake(; config=nothing), x0)
        @test gFD ≈ gZG rtol = 1e-10
        @test gFD ≈ gMC rtol = 1e-10
    end

    @testset "Analytical gradient wrt Acib (linear)" begin
        # d/dA = Σ w_i s1*s2*tmpl_i ; result(A) = A * s1*s2 * tmpl
        s1 = CMBForegrounds.cib_mbb_sed_weight(β, Tdust, ν0_cib, ν1)
        s2 = CMBForegrounds.cib_mbb_sed_weight(β, Tdust, ν0_cib, ν2)
        d_A = sum(w .* (s1 * s2) .* tmpl)
        g = DI.gradient(f, AutoForwardDiff(), x0)
        @test g[1] ≈ d_A rtol = 1e-12
    end
end


@testset "tsz_cib_template_power — type stability + AD" begin

    tmpl   = collect(range(0.1, 0.6, length=100))
    ξ      = 0.10
    A_tSZ  = 5.0
    A_CIB  = 4.0
    β      = 1.75
    Tdust  = 25.0
    ν0_tsz = 143.0
    ν0_cib = 143.0
    ν_sz1, ν_sz2     = 100.24, 222.044
    ν_cib1, ν_cib2   = 105.2, 228.1

    w = randn(MersenneTwister(12), length(tmpl))

    # Loss over [ξ, A_tSZ, A_CIB, β]
    f = x -> sum(w .* CMBForegrounds.tsz_cib_template_power(
        tmpl, x[1], x[2], x[3], x[4], Tdust,
        ν0_tsz, ν0_cib, ν_sz1, ν_sz2, ν_cib1, ν_cib2))
    x0 = [ξ, A_tSZ, A_CIB, β]

    @testset "JET @test_opt — Float64" begin
        JET.@test_opt CMBForegrounds.tsz_cib_template_power(
            tmpl, ξ, A_tSZ, A_CIB, β, Tdust,
            ν0_tsz, ν0_cib, ν_sz1, ν_sz2, ν_cib1, ν_cib2)
    end

    @testset "JET @test_opt — Dual β" begin
        β_d = ForwardDiff.Dual(β, 1.0)
        JET.@test_opt target_modules=(CMBForegrounds,) CMBForegrounds.tsz_cib_template_power(
            tmpl, ξ, A_tSZ, A_CIB, β_d, Tdust,
            ν0_tsz, ν0_cib, ν_sz1, ν_sz2, ν_cib1, ν_cib2)
    end

    @testset "DI gradient — ForwardDiff" begin
        g = DI.gradient(f, AutoForwardDiff(), x0)
        @test length(g) == 4
        @test all(isfinite, g)
    end

    @testset "DI gradient — Zygote" begin
        g = DI.gradient(f, AutoZygote(), x0)
        @test all(isfinite, g)
    end

    @testset "DI gradient — Mooncake" begin
        g = DI.gradient(f, AutoMooncake(; config=nothing), x0)
        @test all(isfinite, g)
    end

    @testset "Cross-backend agreement" begin
        gFD = DI.gradient(f, AutoForwardDiff(),              x0)
        gZG = DI.gradient(f, AutoZygote(),                   x0)
        gMC = DI.gradient(f, AutoMooncake(; config=nothing), x0)
        @test gFD ≈ gZG rtol = 1e-10
        @test gFD ≈ gMC rtol = 1e-10
    end

    @testset "Analytical gradient wrt ξ (linear)" begin
        # result(ξ) = -ξ * sqrt(|A_CIB · A_tSZ|) * (g1·s2 + g2·s1) · tmpl
        # d/dξ      = -sqrt(|A_CIB · A_tSZ|) * (g1·s2 + g2·s1) · tmpl
        g1 = CMBForegrounds.tsz_g_ratio(ν_sz1, ν0_tsz, CMBForegrounds.T_CMB)
        g2 = CMBForegrounds.tsz_g_ratio(ν_sz2, ν0_tsz, CMBForegrounds.T_CMB)
        s1 = CMBForegrounds.cib_mbb_sed_weight(β, Tdust, ν0_cib, ν_cib1)
        s2 = CMBForegrounds.cib_mbb_sed_weight(β, Tdust, ν0_cib, ν_cib2)
        dResult_dξ = -sqrt(abs(A_CIB * A_tSZ)) * (g1 * s2 + g2 * s1) .* tmpl
        d_ξ = sum(w .* dResult_dξ)
        g = DI.gradient(f, AutoForwardDiff(), x0)
        @test g[1] ≈ d_ξ rtol = 1e-12
    end

    @testset "Analytical gradient wrt β (non-linear)" begin
        # result(β) = -ξ · √(|A_CIB·A_tSZ|) · (g1·s2(β) + g2·s1(β)) · tmpl
        # ∂result/∂β computed by ForwardDiff on cib_mbb_sed_weight, which
        # we reuse here as an independent reference.
        g1 = CMBForegrounds.tsz_g_ratio(ν_sz1, ν0_tsz, CMBForegrounds.T_CMB)
        g2 = CMBForegrounds.tsz_g_ratio(ν_sz2, ν0_tsz, CMBForegrounds.T_CMB)
        ds1_dβ = ForwardDiff.derivative(b -> CMBForegrounds.cib_mbb_sed_weight(b, Tdust, ν0_cib, ν_cib1), β)
        ds2_dβ = ForwardDiff.derivative(b -> CMBForegrounds.cib_mbb_sed_weight(b, Tdust, ν0_cib, ν_cib2), β)
        prefac = -ξ * sqrt(abs(A_CIB * A_tSZ))
        dResult_dβ = prefac * (g1 * ds2_dβ + g2 * ds1_dβ) .* tmpl
        d_β = sum(w .* dResult_dβ)
        g = DI.gradient(f, AutoForwardDiff(), x0)
        @test g[4] ≈ d_β rtol = 1e-10
    end

    @testset "Gradient remains finite for small amplitudes" begin
        # Hard test for the sqrt(abs(...)) guard: HMC routinely visits
        # parameter space near A = 0. With A_CIB = 1e-10 the pullback
        # value 1/(2√A) ≈ 5e4 is still finite; without the abs-guard a
        # negative A produces DomainError.
        x_small = [ξ, A_tSZ, 1e-10, β]
        gFD = DI.gradient(f, AutoForwardDiff(), x_small)
        @test all(isfinite, gFD)
    end

    @testset "Negative ξ does not crash and flips sign of factor" begin
        # The model docstring marks ξ ≥ 0, but a leapfrog step can
        # propose ξ < 0 transiently. The function must return finite
        # values; the sign of the contribution should flip relative to
        # ξ > 0 with all else equal.
        v_pos = CMBForegrounds.tsz_cib_template_power(
            tmpl,  0.1, A_tSZ, A_CIB, β, Tdust,
            ν0_tsz, ν0_cib, ν_sz1, ν_sz2, ν_cib1, ν_cib2)
        v_neg = CMBForegrounds.tsz_cib_template_power(
            tmpl, -0.1, A_tSZ, A_CIB, β, Tdust,
            ν0_tsz, ν0_cib, ν_sz1, ν_sz2, ν_cib1, ν_cib2)
        @test all(isfinite, v_pos)
        @test all(isfinite, v_neg)
        @test all(v_neg .≈ -v_pos)
    end

    @testset "Negative amplitude does not crash (sqrt-abs guard)" begin
        # A_tSZ < 0 is unphysical but sampler can propose it transiently.
        # With sqrt(abs(...)), the function is well-defined — no DomainError.
        v_neg_amp = CMBForegrounds.tsz_cib_template_power(
            tmpl, ξ, -1.0, A_CIB, β, Tdust,
            ν0_tsz, ν0_cib, ν_sz1, ν_sz2, ν_cib1, ν_cib2)
        @test all(isfinite, v_neg_amp)
    end
end


@testset "Float32 input behavior (documents widening)" begin
    # The SED helpers always widen Float32 → Float64 because the
    # transcendental kernel `dimensionless_freq_vars` multiplies by the
    # Float64 module constant `Ghz_Kelvin = h/k_B`, and `cib_mbb_sed_weight`
    # additionally promotes against the Float64 default `T_CMB`. The
    # template-based helpers inherit this. Even passing every input
    # (template, amplitudes, β, frequencies, T_CMB) as Float32 yields a
    # Float64 result. The test pins the current behavior so a future
    # change (toward proper Float32 preservation, which would require
    # parameterizing `Ghz_Kelvin`) is caught.
    tmpl_f32 = Float32.(collect(range(0.5f0, 1.5f0, length=10)))
    out_clust = CMBForegrounds.cib_clustered_template_power(
        tmpl_f32, 4.0f0, 1.75f0, 25.0f0, 143.0f0, 147.5f0, 228.1f0)
    @test eltype(out_clust) === Float64

    out_szxcib = CMBForegrounds.tsz_cib_template_power(
        tmpl_f32, 0.1f0, 5.0f0, 4.0f0, 1.75f0, 25.0f0,
        143.0f0, 143.0f0, 100.24f0, 222.044f0, 105.2f0, 228.1f0)
    @test eltype(out_szxcib) === Float64

    # Even with an explicit Float32 T_CMB, widening still happens via
    # `Ghz_Kelvin`. Pin this too — surprising users want a clear failure
    # before their MCMC runs at unintended Float64.
    out_f32_attempt = CMBForegrounds.cib_clustered_template_power(
        tmpl_f32, 4.0f0, 1.75f0, 25.0f0, 143.0f0, 147.5f0, 228.1f0;
        T_CMB=Float32(CMBForegrounds.T_CMB))
    @test eltype(out_f32_attempt) === Float64
end
