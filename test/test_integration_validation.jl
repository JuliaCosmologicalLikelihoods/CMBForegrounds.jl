using Test
using CMBForegrounds
using LinearAlgebra
using ForwardDiff
using Mooncake
using Zygote
using DifferentiationInterface
const DI = DifferentiationInterface
using JET
using Random

@testset "Survey-style composition smoke tests" begin
    rng = MersenneTwister(42)
    ells = unique(round.(Int, range(100, 3000, length=50)))
    n_ell = length(ells)

    # Synthetic dense integer-grid templates. These are composition tests, not
    # released survey reference vectors.
    template_ells = 0:3000
    tsz_template = [1.0 / (1.0 + (l / 3000)^2) for l in template_ells]
    cibc_template = [(l / 3000)^0.8 for l in template_ells]
    szxcib_template = [(l / 3000)^0.5 for l in template_ells]
    dust_template = [(3000 / max(l, 1))^0.6 for l in template_ells]

    # ----------------------------------------------------------------- #
    # 1. HiLLiPoP-like configuration                                    #
    # ----------------------------------------------------------------- #
    @testset "1. HiLLiPoP-like configuration" begin
        # Channels: 100, 143, 217, 353 GHz with effective single frequencies
        bands = [point_band(100.0), point_band(143.0), point_band(217.0), point_band(353.0)]
        n_freq = length(bands)

        # Components
        # Dust: 353 GHz pivot, MBB SED + template
        dust_sed = ModifiedBlackbodySED(353.0, 19.6)
        dust_comp = SkyComponent(dust_sed, TemplateShape(dust_template, 3000.0))
        D_dust = eval_component(dust_comp, ells, bands, 15.0, 1.55)
        @test size(D_dust) == (n_freq, n_freq, n_ell)

        # Radio point sources: 143 GHz pivot, flux convention, Poisson
        radio_sed_model = RadioSED(143.0; convention=:flux)
        radio_comp = SkyComponent(radio_sed_model, PoissonShape(3000.0))
        D_radio = eval_component(radio_comp, ells, bands, 8.0, -0.7)
        @test size(D_radio) == (n_freq, n_freq, n_ell)

        # Correlated tSZ x CIB
        tsz_sed_model = ThermalSZSED(143.0)
        cib_sed_model = ModifiedBlackbodySED(150.0, 25.0)
        cross_corr = TemplateCorrelation(szxcib_template)
        f_tsz = [sed_weight(tsz_sed_model, b) for b in bands]
        f_cib = [sed_weight(cib_sed_model, b, 1.75) for b in bands]
        # Symmetrized pair cross spectrum
        D_szxcib = [correlation_power(cross_corr, ells, -0.2, 4.0, 6.0,
                                      f_tsz[i], f_tsz[j], f_cib[i], f_cib[j])
                    for i in 1:n_freq, j in 1:n_freq]
        @test length(D_szxcib) == n_freq * n_freq
        @test all(D_szxcib[i, j] ≈ D_szxcib[j, i]
                  for i in 1:n_freq, j in 1:n_freq)
        D_total = D_dust .+ D_radio
        for i in 1:n_freq, j in 1:n_freq
            D_total[i, j, :] .+= D_szxcib[i, j]
        end
        @test all(isfinite, D_total)

        # AD verification w.r.t dust amplitude
        loss_hillipop = A -> sum(eval_component(dust_comp, ells, bands, A[1], 1.55))
        g_fd = DI.gradient(loss_hillipop, AutoForwardDiff(), [15.0])
        g_mc = DI.gradient(loss_hillipop, AutoMooncake(config=nothing), [15.0])
        @test isapprox(g_fd, g_mc; rtol=1e-6)
    end

    # ----------------------------------------------------------------- #
    # 2. SPT2018-like configuration                                     #
    # ----------------------------------------------------------------- #
    @testset "2. SPT2018-like configuration" begin
        # Channels: 95, 150, 220 GHz with DeltaBand
        bands = [DeltaBand(95.0), DeltaBand(150.0), DeltaBand(220.0)]
        n_freq = length(bands)

        # Components
        # Poisson radio & DSFG
        radio_comp = SkyComponent(RadioSED(150.0; convention=:flux), PoissonShape(3000.0))
        dsfg_comp = SkyComponent(ModifiedBlackbodySED(150.0, 25.0), PoissonShape(3000.0))
        D_radio = eval_component(radio_comp, ells, bands, 2.0, -0.7)
        D_dsfg = eval_component(dsfg_comp, ells, bands, 7.0, 1.75)

        # Clustered CIB
        cibc_comp = SkyComponent(ModifiedBlackbodySED(150.0, 25.0), PowerLawShape(3000.0))
        D_cibc = eval_component(cibc_comp, ells, bands, 5.0, 1.75; alpha=0.8)

        # tSZ
        tsz_comp = SkyComponent(ThermalSZSED(143.0), TemplateShape(tsz_template, 3000.0))
        D_tsz = eval_component(tsz_comp, ells, bands, 4.0)

        # Geometric-mean correlation tSZ x CIB
        geom_corr = GeometricMeanCorrelation()
        D_szxcib = [correlation_power(geom_corr, tsz_comp, cibc_comp, ells,
                                      bands[i].nu_eff, bands[j].nu_eff,
                                      0.15, 4.0, 5.0, (), (1.75,);
                                      comp2_angular_args=(alpha=0.8,))
                    for i in 1:n_freq, j in 1:n_freq]

        # Total model spectrum
        D_tot = D_radio .+ D_dsfg .+ D_cibc .+ D_tsz
        for i in 1:n_freq, j in 1:n_freq
            D_tot[i, j, :] .+= D_szxcib[i, j]
        end

        # Beam eigenmode perturbation
        modes = [0.01 * (l / 1000.0) for l in ells, k in 1:2]
        coeffs = [0.1, -0.05]
        D_beam_pert = beam_eigenmode_response(D_tot[2, 2, :], modes, coeffs; linearized=false)
        @test all(isfinite, D_beam_pert)
        D_with_beam = copy(D_tot)
        D_with_beam[2, 2, :] = D_beam_pert

        # Map calibration
        gains = [1.01, 1.00, 0.99]
        D_cal = apply_calibration(D_with_beam, gains; convention=:forward)
        @test size(D_cal) == (n_freq, n_freq, n_ell)

        # AD verification
        loss_spt = g -> sum(apply_calibration(D_tot, g; convention=:forward))
        g_fd = DI.gradient(loss_spt, AutoForwardDiff(), gains)
        g_mc = DI.gradient(loss_spt, AutoMooncake(config=nothing), gains)
        @test isapprox(g_fd, g_mc; rtol=1e-6)
    end

    # ----------------------------------------------------------------- #
    # 3. ACT-like configuration                                         #
    # ----------------------------------------------------------------- #
    @testset "3. ACT-like configuration" begin
        # 3 channels with tabulated passbands and chromatic beams
        nu_grid = collect(range(80.0, 240.0, length=41))
        bp1 = exp.(-0.5 .* ((nu_grid .- 95.0) ./ 8.0).^2)
        bp2 = exp.(-0.5 .* ((nu_grid .- 150.0) ./ 10.0).^2)
        bp3 = exp.(-0.5 .* ((nu_grid .- 220.0) ./ 12.0).^2)
        band1 = make_band(nu_grid, bp1)
        band2 = make_band(nu_grid, bp2)
        band3 = make_band(nu_grid, bp3)
        bands = [band1, band2, band3]
        n_freq = 3

        # Chromatic beams
        b_mats = [[1.0 + 0.02 * (n - 150.0)/50.0 * (l / 2000.0) for l in ells, n in nu_grid] for _ in 1:3]
        chrom_beams = [ChromaticBeam(ells, b_mats[i]) for i in 1:3]

        # Components with chromatic beams
        # tSZ with tilt
        tsz_comp = SkyComponent(ThermalSZSED(143.0), TiltedTemplateShape(tsz_template, 3000.0))
        D_tsz = eval_component(tsz_comp, ells, bands, chrom_beams, 4.5; alpha=-0.1)

        # CIB clustered
        cibc_comp = SkyComponent(ModifiedBlackbodySED(150.0, 25.0), TemplateShape(cibc_template, 3000.0))
        D_cibc = eval_component(cibc_comp, ells, bands, chrom_beams, 6.0, 1.75)

        # Correlated cross
        cross_corr = TemplateCorrelation(szxcib_template)
        f_tsz = [sed_weight(tsz_comp.sed, bands[i], chrom_beams[i]) for i in 1:n_freq]
        f_cib = [sed_weight(cibc_comp.sed, bands[i], chrom_beams[i], 1.75) for i in 1:n_freq]
        D_szxcib = [begin
            correlation_power(cross_corr, ells, 0.1, 4.5, 6.0,
                              f_tsz[i], f_tsz[j], f_cib[i], f_cib[j])
        end for i in 1:n_freq, j in 1:n_freq]
        @test all(D_szxcib[i, j] ≈ D_szxcib[j, i]
                  for i in 1:n_freq, j in 1:n_freq)

        D_tot = D_tsz .+ D_cibc
        for i in 1:n_freq, j in 1:n_freq
            D_tot[i, j, :] .+= D_szxcib[i, j]
        end

        # Map gains
        gains = [1.02, 1.00, 0.98]
        D_act = apply_calibration(D_tot, gains; convention=:forward)
        @test size(D_act) == (n_freq, n_freq, n_ell)
        @test all(isfinite, D_act)

        # AD on calibration
        loss_act = g -> sum(apply_calibration(D_tot, g; convention=:forward))
        g_fd = DI.gradient(loss_act, AutoForwardDiff(), gains)
        g_mc = DI.gradient(loss_act, AutoMooncake(config=nothing), gains)
        @test isapprox(g_fd, g_mc; rtol=1e-6)

        # Differentiate a chromatic component, rather than only fixed-spectrum gains.
        loss_cib(p) = sum(eval_component(cibc_comp, ells, bands, chrom_beams,
                                         p[1], p[2]))
        p_cib = [6.0, 1.75]
        g_cib_fd = DI.gradient(loss_cib, AutoForwardDiff(), p_cib)
        g_cib_mc = DI.gradient(loss_cib, AutoMooncake(config=nothing), p_cib)
        @test all(isfinite, g_cib_fd)
        @test g_cib_fd ≈ g_cib_mc rtol=1e-6
    end

    # ----------------------------------------------------------------- #
    # 4. Plik-like configuration                                        #
    # ----------------------------------------------------------------- #
    @testset "4. Plik-like configuration" begin
        # 3 frequencies: 100, 143, 217 GHz
        bands = [point_band(100.0), point_band(143.0), point_band(217.0)]
        n_freq = 3

        # Dust power law
        dust_comp = SkyComponent(ModifiedBlackbodySED(545.0, 19.6), PowerLawShape(500.0))
        D_dust = eval_component(dust_comp, ells, bands, 20.0, 1.51; alpha=-2.63)

        # Calibration with inverse convention (data calibrated or theory / c_i c_j)
        gains = [1.001, 1.000, 0.998]
        D_cal_inv = apply_calibration(D_dust, gains; convention=:inverse)
        for i in 1:3, j in 1:3
            @test D_cal_inv[i, j, :] ≈ D_dust[i, j, :] ./ (gains[i] * gains[j])
        end

        # Polarization leakage
        gammas = [0.01, 0.02, 0.015]
        D_TE = 0.5 .* D_dust
        D_TT = 2.0 .* D_dust
        D_TE_leak = apply_te_leakage(D_TE, D_TT, gammas)
        for i in 1:3, j in 1:3
            @test D_TE_leak[i, j, :] ≈ D_TE[i, j, :] .+ gammas[j] .* D_TT[i, j, :]
        end

        # Aberration
        D_ab = apply_aberration(ells, 0.001, D_dust[2, 2, :])
        @test length(D_ab) == n_ell
        @test all(isfinite, D_ab)
    end

    # ----------------------------------------------------------------- #
    # 5. CamSpec-like configuration                                     #
    # ----------------------------------------------------------------- #
    @testset "5. CamSpec-like configuration" begin
        # Spectrum-independent calibration matrix
        G_mat = [1.00 1.02; 1.02 0.99]
        D = ones(2, 2, n_ell)
        D_cal = apply_calibration(D, G_mat; convention=:forward)
        for i in 1:2, j in 1:2
            @test D_cal[i, j, :] ≈ fill(G_mat[i, j], n_ell)
        end

        # CamSpec residual power laws without physical SED
        camspec_comp = SkyComponent(NoSED(), PowerLawShape(1500.0))
        D_res = eval_component(camspec_comp, ells, [point_band(100.0), point_band(143.0)], 12.0; alpha=-0.8)
        @test size(D_res) == (2, 2, n_ell)
        @test all(isfinite, D_res)
    end

    # ----------------------------------------------------------------- #
    # 6. Assembler regression                                           #
    # ----------------------------------------------------------------- #
    @testset "6. Assembler regression" begin
        n_f = 6
        n_l = 40
        rng_b = MersenneTwister(123)
        ap, ag, as = 0.5, 0.7, 0.3
        fk   = rand(rng_b, n_f); fcp  = rand(rng_b, n_f)
        fd   = rand(rng_b, n_f); fr   = rand(rng_b, n_f)
        ft   = rand(rng_b, n_f); fc   = rand(rng_b, n_f)
        ck   = rand(rng_b, n_l); ccp  = rand(rng_b, n_l)
        cdt  = rand(rng_b, n_l); crd  = rand(rng_b, n_l)
        ct   = rand(rng_b, n_l); cc   = rand(rng_b, n_l)
        csxc = rand(rng_b, n_l)

        # Baseline assembler calls
        D_TT = assemble_TT(ap, ag, as, fk, fcp, fd, fr, ft, fc, ck, ccp, cdt, crd, ct, cc, csxc)
        @test size(D_TT) == (n_f, n_f, n_l)
        @test all(isfinite, D_TT)

        D_EE = assemble_EE(ag, as, fd, fr, cdt, crd)
        @test size(D_EE) == (n_f, n_f, n_l)
        @test all(isfinite, D_EE)

        D_TE = assemble_TE(ag, as, fd, fd, fr, fr, cdt, crd)
        @test size(D_TE) == (n_f, n_f, n_l)
        @test all(isfinite, D_TE)

        # Compare the fused implementation with independent composition.
        D_TT_ref = factorized_cross(fk, ck) .+
                   ap .* factorized_cross(fcp, ccp) .+
                   ag .* factorized_cross(fd, cdt) .+
                   as .* factorized_cross(fr, crd) .+
                   factorized_cross(ft, ct) .+
                   factorized_cross(fc, cc) .+
                   factorized_cross_te(ft, fc, csxc) .+
                   factorized_cross_te(fc, ft, csxc)
        @test D_TT ≈ D_TT_ref rtol=1e-14

        # Prepared Mooncake gradient verification
        v0 = vcat([ap, ag, as], fk, fcp, fd, fr, ft, fc, ck, ccp, cdt, crd, ct, cc, csxc)
        function loss_TT(v)
            _ap, _ag, _as = v[1], v[2], v[3]
            idx = 4
            _fk   = v[idx:idx+n_f-1]; idx += n_f
            _fcp  = v[idx:idx+n_f-1]; idx += n_f
            _fd   = v[idx:idx+n_f-1]; idx += n_f
            _fr   = v[idx:idx+n_f-1]; idx += n_f
            _ft   = v[idx:idx+n_f-1]; idx += n_f
            _fc   = v[idx:idx+n_f-1]; idx += n_f
            _ck   = v[idx:idx+n_l-1]; idx += n_l
            _ccp  = v[idx:idx+n_l-1]; idx += n_l
            _cdt  = v[idx:idx+n_l-1]; idx += n_l
            _crd  = v[idx:idx+n_l-1]; idx += n_l
            _ct   = v[idx:idx+n_l-1]; idx += n_l
            _cc   = v[idx:idx+n_l-1]; idx += n_l
            _csxc = v[idx:idx+n_l-1]
            return sum(assemble_TT(_ap, _ag, _as, _fk, _fcp, _fd, _fr, _ft, _fc, _ck, _ccp, _cdt, _crd, _ct, _cc, _csxc))
        end

        prep_mc = DI.prepare_gradient(loss_TT, AutoMooncake(; config=nothing), v0)
        grad_out = similar(v0)
        DI.gradient!(loss_TT, grad_out, prep_mc, AutoMooncake(; config=nothing), v0)
        grad_fd = DI.gradient(loss_TT, AutoForwardDiff(), v0)
        @test grad_out ≈ grad_fd rtol=1e-10
        @test length(grad_out) == length(v0)
    end
end
