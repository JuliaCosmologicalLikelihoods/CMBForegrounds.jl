"""
    test_correlation_dispatch.jl

Tests for the correlation representation family:
  TemplateCorrelation, GeometricMeanCorrelation, and correlation_power.
"""

using Test
using CMBForegrounds
using JET
using ADTypes
import DifferentiationInterface as DI
using ForwardDiff
using Mooncake
using Zygote
using Random

@testset "Correlation Dispatch Interface" begin
    rng = MersenneTwister(999)
    ells = collect(2:1000)

    # Physics parameters
    A_tSZ = 3.3
    A_CIB = 5.2
    xi = 0.15
    beta_cib = 1.75
    T_dust = 25.0
    nu0_sz = 143.0
    nu0_cib = 150.0

    raw_template = abs.(randn(rng, 3501)) .+ 0.1

    # ----------------------------------------------------------------- #
    # 1. TemplateCorrelation (Signed Cross Model)                       #
    # ----------------------------------------------------------------- #
    @testset "TemplateCorrelation" begin
        corr = TemplateCorrelation(raw_template)

        # 1a. Equivalence with tsz_cib_template_power at positive amplitudes
        nu1 = 150.0
        nu2 = 220.0
        g1 = tsz_g_ratio(nu1, nu0_sz, CMBForegrounds.T_CMB)
        g2 = tsz_g_ratio(nu2, nu0_sz, CMBForegrounds.T_CMB)
        s1 = cib_mbb_sed_weight(beta_cib, T_dust, nu0_cib, nu1)
        s2 = cib_mbb_sed_weight(beta_cib, T_dust, nu0_cib, nu2)

        D_corr = correlation_power(corr, ells, xi, A_tSZ, A_CIB, g1, g2, s1, s2)
        D_ref = tsz_cib_template_power(raw_template[ells .+ 1], xi, A_tSZ, A_CIB,
                                       beta_cib, T_dust, nu0_sz, nu0_cib,
                                       nu1, nu2, nu1, nu2)
        @test D_corr ≈ D_ref

        # 1b. Symmetrized factor of 2 when channel 1 == channel 2
        g_same = tsz_g_ratio(150.0, nu0_sz, CMBForegrounds.T_CMB)
        s_same = cib_mbb_sed_weight(beta_cib, T_dust, nu0_cib, 150.0)
        D_same = correlation_power(corr, ells, xi, A_tSZ, A_CIB, g_same, g_same, s_same, s_same)
        # Factor is: -xi * sqrt(A_CIB*A_tSZ) * (2 * g_same * s_same) * template
        expected_factor = -xi * sqrt(A_CIB * A_tSZ) * (2 * g_same * s_same)
        @test D_same ≈ expected_factor .* raw_template[ells .+ 1]

        # 1c. Channel swap symmetry: (1, 2) == (2, 1)
        D_swap = correlation_power(corr, ells, xi, A_tSZ, A_CIB, g2, g1, s2, s1)
        @test D_corr ≈ D_swap

        # 1d. tSZ frequencies on both sides of the null (ν_null ≈ 217.4 GHz)
        # Physical unnormalized tSZ spectral function:
        @test tsz_f(150.0) < 0
        @test tsz_f(353.0) > 0

        # Normalized tSZ SED (relative to 143 GHz where tsz_f < 0):
        # Therefore tsz_sed > 0 below null and tsz_sed < 0 above null
        g_150 = tsz_g_ratio(150.0, nu0_sz, CMBForegrounds.T_CMB)
        g_353 = tsz_g_ratio(353.0, nu0_sz, CMBForegrounds.T_CMB)
        s_150 = cib_mbb_sed_weight(beta_cib, T_dust, nu0_cib, 150.0)
        s_353 = cib_mbb_sed_weight(beta_cib, T_dust, nu0_cib, 353.0)

        @test g_150 > 0
        @test g_353 < 0

        # Below null for both channels: (g150*s150 + g150*s150) > 0 => with -xi => NEGATIVE
        D_below = correlation_power(corr, ells, xi, A_tSZ, A_CIB, g_150, g_150, s_150, s_150)
        @test all(D_below .< 0)

        # Above null for both channels: (g353*s353 + g353*s353) < 0 => with -xi => POSITIVE
        D_above = correlation_power(corr, ells, xi, A_tSZ, A_CIB, g_353, g_353, s_353, s_353)
        @test all(D_above .> 0)

        # 1e. Zero limits
        @test all(correlation_power(corr, ells, 0.0, A_tSZ, A_CIB, g1, g2, s1, s2) .== 0.0)
        @test all(correlation_power(corr, ells, xi, 0.0, A_CIB, g1, g2, s1, s2) .== 0.0)
        @test all(correlation_power(corr, ells, xi, A_tSZ, 0.0, g1, g2, s1, s2) .== 0.0)

        # 1f. Type stability
        JET.@test_opt correlation_power(corr, ells, xi, A_tSZ, A_CIB, g1, g2, s1, s2)

        # 1g. AD wrt xi, A_tSZ, A_CIB
        g_ad(p) = sum(correlation_power(corr, ells, p[1], p[2], p[3], g1, g2, s1, s2))
        p0 = [xi, A_tSZ, A_CIB]
        grad_fd = DI.gradient(g_ad, AutoForwardDiff(), p0)
        grad_mk = DI.gradient(g_ad, AutoMooncake(; config=nothing), p0)
        grad_zg = DI.gradient(g_ad, AutoZygote(), p0)
        @test grad_fd ≈ grad_mk rtol=1e-8
        @test grad_fd ≈ grad_zg rtol=1e-8

        corr_power = TemplateCorrelation(PowerLawShape(3000.0))
        power_ells = [1500, 3000]
        D_power = correlation_power(corr_power, power_ells, xi, 2.0, 3.0,
                                    1.0, 1.0, 1.0, 1.0; alpha=0.8)
        expected_power = @. -2xi * sqrt(6.0) * (power_ells / 3000)^0.8
        @test D_power ≈ expected_power

        normalized = TemplateShape(2 .* ones(3001); ell_0=3000)
        corr_tilted = TemplateCorrelation(TiltedTemplateShape(normalized, 3000.0))
        tilt_loss(p) = sum(correlation_power(
            corr_tilted, power_ells, xi, 2.0, 3.0,
            1.0, 1.0, 1.0, 1.0; alpha=p[1]
        ))
        for alpha in (0.0, 0.8)
            p = [alpha]
            g_fd = DI.gradient(tilt_loss, AutoForwardDiff(), p)
            @test g_fd ≈ DI.gradient(tilt_loss, AutoMooncake(config=nothing), p) rtol=1e-8
            @test g_fd ≈ DI.gradient(tilt_loss, AutoZygote(), p) rtol=1e-8
        end

        @test_throws MethodError correlation_power(
            corr_power, xi, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0
        )
        @test_throws DimensionMismatch correlation_power(
            corr, [100, 200], xi, 2.0, 3.0,
            ones(1), ones(2), ones(2), ones(2)
        )
    end

    # ----------------------------------------------------------------- #
    # 2. GeometricMeanCorrelation (SPT-style Model)                     #
    # ----------------------------------------------------------------- #
    @testset "GeometricMeanCorrelation" begin
        corr_geo = GeometricMeanCorrelation()

        # Autos
        comp_sz = SkyComponent(ThermalSZSED(143.0), TemplateShape(raw_template; ell_0=3000))
        comp_cib = SkyComponent(ModifiedBlackbodySED(150.0, 25.0), PowerLawShape(3000.0))

        nu1 = 95.0
        nu2 = 150.0

        # Component-based correlation_power
        D_geo = correlation_power(corr_geo, comp_sz, comp_cib, ells, nu1, nu2, xi, A_tSZ, A_CIB,
                                  (), (beta_cib,); comp2_angular_args=(alpha=0.8,))

        # Reference via tsz_cib_cross_power
        ref_geo = tsz_cib_cross_power(ells, xi, A_tSZ, A_CIB, 0.8, beta_cib,
                                      1.0, 1.0, nu1, nu2, nu1, nu2,
                                      0.0, raw_template[ells .+ 1] ./ raw_template[3001],
                                      143.0, 25.0, 150.0;
                                      ℓ_pivot_cib=3000, ℓ_pivot_tsz=3000)
        @test D_geo ≈ ref_geo

        # Symmetrized factor of 2 when nu1 == nu2
        D_geo_auto = correlation_power(corr_geo, comp_sz, comp_cib, ells, nu1, nu1, xi, A_tSZ, A_CIB,
                                       (), (beta_cib,); comp2_angular_args=(alpha=0.8,))
        ref_geo_auto = tsz_cib_cross_power(ells, xi, A_tSZ, A_CIB, 0.8, beta_cib,
                                           1.0, 1.0, nu1, nu1, nu1, nu1,
                                           0.0, raw_template[ells .+ 1] ./ raw_template[3001],
                                           143.0, 25.0, 150.0;
                                           ℓ_pivot_cib=3000, ℓ_pivot_tsz=3000)
        @test D_geo_auto ≈ ref_geo_auto

        # Channel swap symmetry
        D_geo_swap = correlation_power(corr_geo, comp_sz, comp_cib, ells, nu2, nu1, xi, A_tSZ, A_CIB,
                                       (), (beta_cib,); comp2_angular_args=(alpha=0.8,))
        @test D_geo ≈ D_geo_swap

        # Direct evaluation from pre-computed autos
        D1_11 = eval_component(comp_sz, ells, nu1, nu1, A_tSZ)
        D1_22 = eval_component(comp_sz, ells, nu2, nu2, A_tSZ)
        D2_11 = eval_component(comp_cib, ells, nu1, nu1, A_CIB, beta_cib; alpha=0.8)
        D2_22 = eval_component(comp_cib, ells, nu2, nu2, A_CIB, beta_cib; alpha=0.8)
        D_direct = correlation_power(corr_geo, ells, xi, D1_11, D1_22, D2_11, D2_22)
        @test D_direct ≈ D_geo

        @test_throws DimensionMismatch correlation_power(
            corr_geo, [100, 200, 300], xi,
            ones(1), ones(1), ones(1), ones(1)
        )

        # Type stability
        JET.@test_opt correlation_power(corr_geo, ells, xi, D1_11, D1_22, D2_11, D2_22)

        # AD wrt xi, A_tSZ, A_CIB
        g_ad_geo(p) = sum(correlation_power(corr_geo, ells, p[1],
                                            p[2] .* D1_11, p[2] .* D1_22,
                                            p[3] .* D2_11, p[3] .* D2_22))
        p0 = [xi, 1.0, 1.0]
        grad_fd = DI.gradient(g_ad_geo, AutoForwardDiff(), p0)
        grad_mk = DI.gradient(g_ad_geo, AutoMooncake(; config=nothing), p0)
        grad_zg = DI.gradient(g_ad_geo, AutoZygote(), p0)
        @test grad_fd ≈ grad_mk rtol=1e-8
        @test grad_fd ≈ grad_zg rtol=1e-8
    end
end
