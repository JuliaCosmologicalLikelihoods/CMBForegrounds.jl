"""
    test_sed_dispatch.jl

Tests for the SED representation family:
  ModifiedBlackbodySED, RadioSED, ThermalSZSED, ConstantSED, NoSED, SkyComponent,
  sed_weight, eval_component, and eval_component_te.
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

struct TestPowerSED <: AbstractSED end
CMBForegrounds.sed_weight(::TestPowerSED, nu::Real, beta::Real) = (nu / 150.0)^beta

@testset "SED Dispatch & Component Composition" begin
    rng = MersenneTwister(12345)
    ells = collect(2:1000)

    # ----------------------------------------------------------------- #
    # 1. ModifiedBlackbodySED                                           #
    # ----------------------------------------------------------------- #
    @testset "ModifiedBlackbodySED" begin
        sed = ModifiedBlackbodySED(150.0, 25.0)
        nu = 220.0
        beta = 1.6

        # Parity with cib_mbb_sed_weight and mbb_sed
        w = sed_weight(sed, nu, beta)
        ref_cib = cib_mbb_sed_weight(beta, 25.0, 150.0, nu)
        ref_act = mbb_sed(nu, 150.0, beta, 25.0)
        @test w ≈ ref_cib
        @test w ≈ ref_act

        # Normalization at pivot frequency
        @test sed_weight(sed, 150.0, beta) ≈ 1.0

        # Vector evaluation
        nus = [90.0, 150.0, 220.0]
        ws = sed_weight(sed, nus, beta)
        @test ws ≈ [sed_weight(sed, n, beta) for n in nus]

        # Band integration
        pband = point_band(220.0)
        @test sed_weight(sed, pband, beta) ≈ w

        raw_nu = collect(130.0:1.0:170.0)
        raw_bp = @. exp(-0.5 * ((raw_nu - 150.0) / 10.0)^2)
        band = make_band(raw_nu, raw_bp)
        w_band = sed_weight(sed, band, beta)
        @test isfinite(w_band)
        @test w_band > 0

        # Type stability
        JET.@test_opt sed_weight(sed, nu, beta)
        JET.@test_opt sed_weight(sed, pband, beta)

        # AD wrt beta and T_dust
        g_mbb(p) = sed_weight(ModifiedBlackbodySED(150.0, p[2]), nu, p[1])
        p0 = [beta, 25.0]
        grad_fd = DI.gradient(g_mbb, AutoForwardDiff(), p0)
        grad_mk = DI.gradient(g_mbb, AutoMooncake(; config=nothing), p0)
        grad_zg = DI.gradient(g_mbb, AutoZygote(), p0)
        @test grad_fd ≈ grad_mk rtol=1e-8
        @test grad_fd ≈ grad_zg rtol=1e-8
    end

    # ----------------------------------------------------------------- #
    # 2. RadioSED (conventions & mapping)                               #
    # ----------------------------------------------------------------- #
    @testset "RadioSED (:rj vs :flux conventions)" begin
        # 2a. :rj convention (matches radio_sed, beta_RJ ≈ -2.7)
        sed_rj = RadioSED(150.0; convention=:rj)
        beta_rj = -2.7
        w_rj = sed_weight(sed_rj, 220.0, beta_rj)
        @test w_rj ≈ radio_sed(220.0, 150.0, beta_rj)
        @test sed_weight(sed_rj, 150.0, beta_rj) ≈ 1.0

        # 2b. :flux convention (matches _radio_sed_ratio, beta_flux ≈ -0.7)
        sed_flux = RadioSED(150.0; convention=:flux)
        beta_flux = -0.7
        w_flux = sed_weight(sed_flux, 220.0, beta_flux)
        # Note: _radio_sed_ratio is defined in CMBForegrounds
        ref_flux = CMBForegrounds._radio_sed_ratio(220.0, 150.0, beta_flux, CMBForegrounds.T_CMB)
        @test w_flux ≈ ref_flux
        @test sed_weight(sed_flux, 150.0, beta_flux) ≈ 1.0

        # 2c. Equivalence when beta_rj = beta_flux - 2
        # Verify exact mathematical relation between the two conventions
        @test w_flux ≈ radio_sed(220.0, 150.0, beta_flux - 2) rtol=1e-12

        # Both conventions honor the stored CMB temperature.
        sed_rj_hot = RadioSED(150.0; convention=:rj, T_CMB=4.0)
        sed_flux_hot = RadioSED(150.0; convention=:flux, T_CMB=4.0)
        @test sed_weight(sed_rj_hot, 220.0, beta_rj) ≈
              radio_sed(220.0, 150.0, beta_rj, 4.0)
        @test sed_weight(sed_flux_hot, 220.0, beta_flux) ≈
              CMBForegrounds._radio_sed_ratio(220.0, 150.0, beta_flux, 4.0)

        # Band integration
        pband = point_band(220.0)
        @test sed_weight(sed_rj, pband, beta_rj) ≈ w_rj
        @test sed_weight(sed_flux, pband, beta_flux) ≈ w_flux

        # Type stability
        JET.@test_opt sed_weight(sed_rj, 220.0, beta_rj)
        JET.@test_opt sed_weight(sed_flux, 220.0, beta_flux)

        # AD wrt beta
        g_rad_rj(b) = sed_weight(sed_rj, 220.0, b[1])
        grad_fd = DI.gradient(g_rad_rj, AutoForwardDiff(), [beta_rj])
        grad_mk = DI.gradient(g_rad_rj, AutoMooncake(; config=nothing), [beta_rj])
        grad_zg = DI.gradient(g_rad_rj, AutoZygote(), [beta_rj])
        @test grad_fd ≈ grad_mk rtol=1e-8
        @test grad_fd ≈ grad_zg rtol=1e-8
    end

    # ----------------------------------------------------------------- #
    # 3. ThermalSZSED                                                   #
    # ----------------------------------------------------------------- #
    @testset "ThermalSZSED" begin
        sed_sz = ThermalSZSED(143.0)
        nu = 220.0
        w = sed_weight(sed_sz, nu)
        @test w ≈ tsz_sed(nu, 143.0)
        @test w ≈ tsz_g_ratio(nu, 143.0, CMBForegrounds.T_CMB)
        @test sed_weight(sed_sz, 143.0) ≈ 1.0

        pband = point_band(220.0)
        @test sed_weight(sed_sz, pband) ≈ w

        # The stored CMB temperature is honored by every tSZ path.
        sed_hot = ThermalSZSED(143.0; T_CMB=4.0)
        @test sed_weight(sed_hot, nu) ≈ tsz_g_ratio(nu, 143.0, 4.0)
        @test sed_weight(sed_hot, [143.0, nu]) ≈ tsz_g_ratio.([143.0, nu], 143.0, 4.0)
        @test sed_weight(sed_hot, pband) ≈ tsz_g_ratio(nu, 143.0, 4.0)

        # Type stability
        JET.@test_opt sed_weight(sed_sz, nu)
        JET.@test_opt sed_weight(sed_sz, pband)
    end

    # ----------------------------------------------------------------- #
    # 4. ConstantSED & NoSED                                            #
    # ----------------------------------------------------------------- #
    @testset "ConstantSED & NoSED" begin
        c_sed = ConstantSED()
        no_sed = NoSED()

        @test sed_weight(c_sed, 150.0) == 1.0
        @test sed_weight(c_sed, point_band(150.0)) == 1.0
        @test all(sed_weight(c_sed, [90.0, 150.0]) .== 1.0)

        @test sed_weight(no_sed, 150.0) == 1.0
        @test sed_weight(no_sed, point_band(150.0)) == 1.0
        @test all(sed_weight(no_sed, [90.0, 150.0]) .== 1.0)

        @test sed_weight(c_sed, DeltaBand(150.0)) == 1.0
        @test sed_weight(no_sed, DeltaBand(150.0)) == 1.0

        beam = ChromaticBeam([2, 3], ones(2, 1))
        @test sed_weight(c_sed, DeltaBand(150.0), beam) == ones(2)
        @test sed_weight(no_sed, DeltaBand(150.0), beam) == ones(2)

        # Dispatch remains unambiguous as new band representations are added.
        @test isempty(Test.detect_ambiguities(CMBForegrounds; recursive=true))
    end

    # ----------------------------------------------------------------- #
    # 5. Component Composition (Power-law vs Template)                  #
    # ----------------------------------------------------------------- #
    @testset "Galactic Dust & CIB: Power law vs Template" begin
        # Dust MBB SED
        dust_sed = ModifiedBlackbodySED(150.0, 19.6)
        beta_dust = 1.55
        amp_dust = 2.8

        # 5a. Power-law dust (e.g. SPT-like)
        # In dust_tt_power_law, alpha_in gives (ℓ/ℓ_pivot)^(alpha_in + 2)
        # Here PowerLawShape(80.0) evaluates (ℓ/80.0)^alpha_eff
        alpha_eff = -0.42
        c_dust_pl = eval_component(dust_sed, PowerLawShape(80.0), ells, 150.0, 220.0,
                                   amp_dust, beta_dust; alpha=alpha_eff)
        ref_dust_pl = dust_tt_power_law(ells, amp_dust, alpha_eff - 2.0, beta_dust, 150.0, 220.0, 19.6, 150.0; ℓ_pivot=80.0)
        @test c_dust_pl ≈ ref_dust_pl

        # 5b. Template dust (e.g. HiLLiPoP / Plik)
        raw_dust_tmpl = abs.(randn(rng, 1001)) .+ 0.1
        dust_tmpl = TemplateShape(raw_dust_tmpl; ell_0=nothing, ell_min=0)
        c_dust_tmpl = eval_component(dust_sed, dust_tmpl, ells, 150.0, 220.0, amp_dust, beta_dust)
        s1 = cib_mbb_sed_weight(beta_dust, 19.6, 150.0, 150.0)
        s2 = cib_mbb_sed_weight(beta_dust, 19.6, 150.0, 220.0)
        ref_dust_tmpl = @. amp_dust * s1 * s2 * raw_dust_tmpl[ells .+ 1]
        @test c_dust_tmpl ≈ ref_dust_tmpl

        # 5c. SkyComponent encapsulation
        comp_pl = SkyComponent(dust_sed, PowerLawShape(80.0))
        @test eval_component(comp_pl, ells, 150.0, 220.0, amp_dust, beta_dust; alpha=alpha_eff) ≈ ref_dust_pl

        comp_tmpl = SkyComponent(dust_sed, dust_tmpl)
        @test eval_component(comp_tmpl, ells, 150.0, 220.0, amp_dust, beta_dust) ≈ ref_dust_tmpl

        b150 = DeltaBand(150.0)
        b220 = point_band(220.0)
        @test eval_component(comp_pl, ells, b150, b220, amp_dust, beta_dust;
                             alpha=alpha_eff) ≈ ref_dust_pl

        finite_band = make_band(collect(140.0:160.0), ones(21))
        pair_band = eval_component(comp_pl, ells, finite_band, b220,
                                   amp_dust, beta_dust; alpha=alpha_eff)
        pair_ref = sed_weight(dust_sed, finite_band, beta_dust) *
                   sed_weight(dust_sed, b220, beta_dust) .*
                   angular_power(comp_pl.angular, ells;
                                 amp=amp_dust, alpha=alpha_eff)
        @test pair_band ≈ pair_ref

        pair_ordered = eval_component(
            dust_sed, dust_sed, comp_pl.angular, ells,
            finite_band, b220, amp_dust,
            (beta_dust,), (beta_dust,); alpha=alpha_eff
        )
        @test pair_ordered ≈ pair_ref
        @test_throws DimensionMismatch eval_component(
            comp_pl, ells, [150.0], fill(220.0, length(ells)),
            amp_dust, beta_dust; alpha=alpha_eff
        )
        @test_throws DimensionMismatch eval_component(
            dust_sed, dust_sed, comp_pl.angular, ells,
            fill(150.0, length(ells)), [220.0], amp_dust,
            (beta_dust,), (beta_dust,); alpha=alpha_eff
        )
    end

    @testset "Custom SED band lifting" begin
        sed = TestPowerSED()
        beta = 1.7
        band = make_band(collect(140.0:160.0), ones(21))
        delta = DeltaBand(150.0)
        beam = ChromaticBeam([100, 200], ones(2, length(band.nu)))

        @test sed_weight(sed, 160.0, beta) ≈ (160 / 150)^beta
        @test sed_weight(sed, delta, beta) == 1.0
        @test sed_weight(sed, band, beta) ≈
              integrate_sed(nu -> (nu / 150)^beta, band)
        @test sed_weight(sed, band, beam, beta) ≈
              fill(sed_weight(sed, band, beta), 2)

        loss(b) = sed_weight(sed, band, b[1])
        p = [beta]
        g_fd = DI.gradient(loss, AutoForwardDiff(), p)
        @test g_fd ≈ DI.gradient(loss, AutoMooncake(config=nothing), p) rtol=1e-8
        @test g_fd ≈ DI.gradient(loss, AutoZygote(), p) rtol=1e-8

        @test_throws ArgumentError make_band(Float64[], Float64[])
        @test_throws DimensionMismatch make_band([140.0, 150.0], [1.0])
        @test_throws ArgumentError make_band([150.0, 140.0], ones(2))
        @test_throws ArgumentError make_band([NaN], [1.0])
        @test_throws ArgumentError make_band([140.0, 150.0], [1.0, Inf])
        @test_throws DomainError make_band([140.0, 150.0], zeros(2))
        @test_throws ArgumentError RawBand(Float64[], Float64[])
        @test_throws DimensionMismatch RawBand([140.0, 150.0], [1.0])
        @test_throws ArgumentError RawBand([150.0, 140.0], ones(2))
        @test_throws ArgumentError eval_sed_bands(identity, AbstractBand[])
        zero_beam = ChromaticBeam([100, 200], zeros(2, length(band.nu)))
        @test_throws DomainError sed_weight(sed, band, zero_beam, beta)
        @test isempty(Test.detect_ambiguities(CMBForegrounds; recursive=true))
    end

    # ----------------------------------------------------------------- #
    # 6. Caller-Selected Shared vs Independent Pair Amplitudes          #
    # ----------------------------------------------------------------- #
    @testset "Caller-selected shared vs pair amplitudes" begin
        # Radio Poisson model:
        radio_sed = RadioSED(150.0; convention=:flux)
        poisson = PoissonShape(3000.0)
        freqs = [95.0, 150.0, 220.0]
        n_f = length(freqs)

        # 6a. Shared amplitude model: A_radio = 1.2
        A_shared = 1.2
        D_shared = zeros(n_f, n_f, length(ells))
        for i in 1:n_f, j in 1:n_f
            D_shared[i, j, :] = eval_component(radio_sed, poisson, ells, freqs[i], freqs[j], A_shared, -0.7)
        end

        # 6b. Independent pair amplitudes (like SPT 2018 Table VIII):
        # A_ij is an arbitrary matrix selected by the likelihood caller
        A_pairs = [1.5 0.8 0.4;
                   0.8 1.1 0.6;
                   0.4 0.6 0.9]
        D_pairs = zeros(n_f, n_f, length(ells))
        for i in 1:n_f, j in 1:n_f
            D_pairs[i, j, :] = eval_component(radio_sed, poisson, ells, freqs[i], freqs[j], A_pairs[i, j], -0.7)
        end

        for i in 1:n_f, j in 1:n_f
            @test D_pairs[i, j, :] ≈ (A_pairs[i, j] / A_shared) .* D_shared[i, j, :]
        end
    end

    # ----------------------------------------------------------------- #
    # 7. CamSpec-Style Residual (No Physical SED)                       #
    # ----------------------------------------------------------------- #
    @testset "CamSpec empirical TT residual" begin
        # D_ℓ^res = A_ij * (ℓ / 1500)^gamma_ij
        residual_shape = PowerLawShape(1500.0)
        A_ij = 3.4
        gamma_ij = 0.5

        # Evaluated through eval_component with NoSED()
        res = eval_component(NoSED(), residual_shape, ells, 143.0, 217.0, A_ij; alpha=gamma_ij)
        ref = @. A_ij * (ells / 1500.0)^gamma_ij
        @test res ≈ ref
    end

    # ----------------------------------------------------------------- #
    # 8. Separate T/E Legs & Ordered TE versus ET                       #
    # ----------------------------------------------------------------- #
    @testset "Ordered TE versus ET" begin
        # Galactic dust in polarization has different parameters/frequencies
        # Leg 1 in T (150 GHz), Leg 2 in E (220 GHz)
        sed_T = ModifiedBlackbodySED(150.0, 19.6)
        sed_E = ModifiedBlackbodySED(150.0, 19.6)
        beta_T = 1.53
        beta_E = 1.65

        shape = PowerLawShape(80.0)
        alpha = -0.4
        amp = 1.5

        # TE: nu1=150 (T), nu2=220 (E)
        D_TE_12 = eval_component(sed_T, sed_E, shape, ells, 150.0, 220.0, amp, (beta_T,), (beta_E,); alpha=alpha)
        # ET: nu1=150 (E), nu2=220 (T)
        D_ET_12 = eval_component(sed_E, sed_T, shape, ells, 150.0, 220.0, amp, (beta_E,), (beta_T,); alpha=alpha)

        # Because beta_T != beta_E and nu1 != nu2, TE and ET cross are NOT identical:
        @test !(D_TE_12 ≈ D_ET_12)

        # But swapping channels: TE(1, 2) == ET(2, 1)
        D_ET_21 = eval_component(sed_E, sed_T, shape, ells, 220.0, 150.0, amp, (beta_E,), (beta_T,); alpha=alpha)
        @test D_TE_12 ≈ D_ET_21

        # Tensor evaluation with eval_component_te
        bands_T = [point_band(150.0), point_band(220.0)]
        bands_E = [point_band(150.0), point_band(220.0)]
        D_tensor = eval_component_te(sed_T, sed_E, shape, ells, bands_T, bands_E, amp, (beta_T,), (beta_E,); alpha=alpha)
        @test size(D_tensor) == (2, 2, length(ells))
        @test D_tensor[1, 2, :] ≈ D_TE_12
        @test D_tensor[2, 1, :] ≈ D_ET_12
    end
end
