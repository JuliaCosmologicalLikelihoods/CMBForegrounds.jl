using Test
using CMBForegrounds
using LinearAlgebra
using ChainRulesCore
using ForwardDiff
using Mooncake
using Zygote
using DifferentiationInterface
const DI = DifferentiationInterface
using JET

@testset "Chromatic bandpass and response" begin
    # Setup test frequencies and bands
    nu = collect(range(130.0, 170.0, length=41))
    bp1 = exp.(-0.5 .* ((nu .- 145.0) ./ 8.0).^2)
    bp2 = exp.(-0.5 .* ((nu .- 155.0) ./ 8.0).^2)
    band1 = make_band(nu, bp1)
    band2 = make_band(nu, bp2)
    bands = [band1, band2]

    ells = collect(range(100.0, 3000.0, length=50))
    n_ell = length(ells)
    n_nu = length(nu)

    # Synthetic chromatic beam with frequency and ell dependence
    # b_ell(nu) = exp(-0.5 * (ell * theta(nu))^2)
    beam_mat1 = [1.0 + 0.05 * (n - 145.0)/20.0 * (l / 2000.0) for l in ells, n in nu]
    beam_mat2 = [1.0 + 0.04 * (n - 155.0)/20.0 * (l / 2000.0) for l in ells, n in nu]
    chrom_beam1 = ChromaticBeam(ells, beam_mat1)
    chrom_beam2 = ChromaticBeam(ells, beam_mat2)
    chrom_beams = [chrom_beam1, chrom_beam2]

    @testset "1. DeltaBand limit" begin
        dband1 = DeltaBand(145.0)
        dband2 = DeltaBand(155.0)
        @test dband1.nu_eff == 145.0
        @test integrate_sed(ν -> ν^2, dband1) == 145.0^2

        mbb = ModifiedBlackbodySED(150.0, 19.6)
        @test isapprox(sed_weight(mbb, dband1, 1.5), sed_weight(mbb, 145.0, 1.5))

        # Chromatic integration for DeltaBand returns fill(sed_fn(nu_eff), n_ell)
        F_delta = integrate_chromatic_sed(ν -> ν^2, dband1, chrom_beam1)
        @test length(F_delta) == n_ell
        @test all(F_delta .== 145.0^2)

        # Monofreq Band limit
        pband = point_band(145.0)
        F_mono = integrate_chromatic_sed(ν -> ν^2, pband, chrom_beam1)
        @test all(F_mono .== 145.0^2)

        manual = Band([100.0], [1.0], 145.0, true)
        @test integrate_sed(identity, manual) == 145.0
        @test integrate_tsz(manual, 143.0) ≈ tsz_sed(145.0, 143.0)
        @test integrate_chromatic_sed(identity,
                                      prepare_chromatic_bandpass(manual, chrom_beam1)) ==
              fill(145.0, n_ell)
    end

    @testset "2. Achromatic limit" begin
        # If beam is independent of nu, it cancels completely in Eq. 31
        achrom_mat = repeat(collect(range(1.0, 0.5, length=n_ell)), 1, n_nu)
        achrom_beam = ChromaticBeam(ells, achrom_mat)

        sed_fn = ν -> cib_mbb_sed_weight(1.75, 19.6, 150.0, ν)
        F_chrom = integrate_chromatic_sed(sed_fn, band1, achrom_beam)
        F_scalar = integrate_sed(sed_fn, band1)

        @test isapprox(F_chrom, fill(F_scalar, n_ell); rtol=1e-12)
    end

    @testset "3. Normalized blackbody limit" begin
        # For f(nu) = 1.0 (CMB / kSZ), numerator == denominator identically
        F_unit1 = integrate_chromatic_sed(ν -> 1.0, band1, chrom_beam1)
        F_unit2 = integrate_chromatic_sed(ν -> 1.0, band2, chrom_beam2)
        @test isapprox(F_unit1, ones(n_ell); rtol=1e-14)
        @test isapprox(F_unit2, ones(n_ell); rtol=1e-14)

        # ConstantSED / NoSED
        @test sed_weight(ConstantSED(), band1, chrom_beam1) == ones(n_ell)
        @test sed_weight(NoSED(), band1, chrom_beam1) == ones(n_ell)
    end

    @testset "Prepared chromatic responses" begin
        prepared1 = prepare_chromatic_bandpass(band1, chrom_beam1)
        prepared2 = prepare_chromatic_bandpass(band2, chrom_beam2)
        prepared = [prepared1, prepared2]
        sed_fn = ν -> cib_mbb_sed_weight(1.75, 19.6, 150.0, ν)

        @test integrate_chromatic_sed(sed_fn, prepared1) ≈
              integrate_chromatic_sed(sed_fn, band1, chrom_beam1)
        @test eval_chromatic_sed_bands(sed_fn, prepared) ≈
              eval_chromatic_sed_bands(sed_fn, bands, chrom_beams)

        delta = DeltaBand(145.0)
        prepared_delta = prepare_chromatic_bandpass(delta, chrom_beam1)
        @test integrate_chromatic_sed(sed_fn, prepared_delta) ==
              integrate_chromatic_sed(sed_fn, delta, chrom_beam1)

        mbb = ModifiedBlackbodySED(150.0, 19.6)
        component = SkyComponent(mbb, PowerLawShape(3000.0))
        @test eval_component(component, ells, prepared, 2.0, 1.75; alpha=-0.6) ≈
              eval_component(component, ells, bands, chrom_beams, 2.0, 1.75;
                             alpha=-0.6)
        @test eval_component_te(mbb, mbb, component.angular, ells,
                                prepared, prepared, 2.0,
                                (1.75,), (1.75,); alpha=-0.6) ≈
              eval_component_te(mbb, mbb, component.angular, ells,
                                bands, chrom_beams, bands, chrom_beams, 2.0,
                                (1.75,), (1.75,); alpha=-0.6)

        beta_loss = p -> sum(sed_weight(mbb, prepared1, p[1]))
        beta = [1.75]
        beta_fd = DI.gradient(beta_loss, AutoForwardDiff(), beta)
        @test beta_fd ≈ DI.gradient(beta_loss, AutoZygote(), beta) rtol=1e-8
        @test beta_fd ≈ DI.gradient(beta_loss, AutoMooncake(config=nothing), beta) rtol=1e-6

        raw = RawBand(nu, bp1)
        shift_loss = p -> begin
            shifted = shift_and_normalize(raw, p[1])
            response = prepare_chromatic_bandpass(shifted, chrom_beam1)
            sum(integrate_chromatic_sed(sed_fn, response))
        end
        shift_fd = DI.gradient(shift_loss, AutoForwardDiff(), [0.0])
        @test shift_fd ≈ DI.gradient(shift_loss, AutoZygote(), [0.0]) rtol=1e-8
        @test shift_fd ≈ DI.gradient(shift_loss, AutoMooncake(config=nothing), [0.0]) rtol=1e-6

        beam_direction = [((n - 150.0) / 20)^2 * (l / 2000.0)
                          for l in ells, n in nu]
        beam_loss = p -> begin
            active_beam = ChromaticBeam(ells, beam_mat1 .+ p[1] .* beam_direction)
            response = prepare_chromatic_bandpass(band1, active_beam)
            sum(integrate_chromatic_sed(sed_fn, response))
        end
        beam_fd = DI.gradient(beam_loss, AutoForwardDiff(), [0.1])
        @test beam_fd ≈ DI.gradient(beam_loss, AutoZygote(), [0.1]) rtol=1e-8
        @test beam_fd ≈ DI.gradient(beam_loss, AutoMooncake(config=nothing), [0.1]) rtol=1e-6

        shifted_response = prepare_chromatic_bandpass(
            shift_and_normalize(raw, 1.0), chrom_beam1
        )
        @test !isapprox(integrate_chromatic_sed(sed_fn, prepared1),
                        integrate_chromatic_sed(sed_fn, shifted_response))

        @test_throws DimensionMismatch prepare_chromatic_bandpass(
            band1, ChromaticBeam(ells, ones(n_ell, n_nu - 1))
        )
        @test_throws ArgumentError eval_chromatic_sed_bands(
            sed_fn, PreparedChromaticBandpass[]
        )
    end

    @testset "4. Shifted ACT reference cases" begin
        raw1 = RawBand(nu, bp1)
        # Shift = 0 matches unshifted make_band
        b_shift0 = shift_and_normalize(raw1, 0.0)
        @test isapprox(b_shift0.norm_bp, band1.norm_bp; rtol=1e-12)

        # Nonzero shift
        shift_val = 1.5 # GHz
        b_shifted = shift_and_normalize(raw1, shift_val)
        @test isapprox(b_shifted.nu, nu .+ shift_val)

        # Chromatic beam calculation with unshifted beam matrix (ACT approximation)
        sed_fn = ν -> cib_mbb_sed_weight(1.75, 19.6, 150.0, ν)
        F_shifted = integrate_chromatic_sed(sed_fn, b_shifted, chrom_beam1)
        @test length(F_shifted) == n_ell
        @test all(isfinite, F_shifted)
        @test !isapprox(F_shifted, integrate_chromatic_sed(sed_fn, band1, chrom_beam1))

        # Differentiability w.r.t shift
        f_shift = s -> begin
            b_s = shift_and_normalize(raw1, s[1])
            sum(integrate_chromatic_sed(sed_fn, b_s, chrom_beam1))
        end
        g_fd = DI.gradient(f_shift, AutoForwardDiff(), [0.0])
        @test isfinite(g_fd[1])

        # A parameter may affect any channel, not only the first one.
        mixed_loss = s -> begin
            shifted = shift_and_normalize(raw1, s[1])
            sum(eval_chromatic_sed_bands(ν -> (ν / 150)^2,
                                         [band1, shifted],
                                         [chrom_beam1, chrom_beam1]))
        end
        mixed_grad = DI.gradient(mixed_loss, AutoForwardDiff(), [0.0])
        @test all(isfinite, mixed_grad)
        mixed_grad_zg = DI.gradient(mixed_loss, AutoZygote(), [0.0])
        mixed_grad_mc = DI.gradient(mixed_loss, AutoMooncake(config=nothing), [0.0])
        @test mixed_grad ≈ mixed_grad_zg rtol=1e-8
        @test mixed_grad ≈ mixed_grad_mc rtol=1e-6

        bad_grid = ChromaticBeam(ells .+ 1, chrom_beam2.beam)
        @test_throws ArgumentError eval_chromatic_sed_bands(sed_fn, bands,
                                                             [chrom_beam1, bad_grid])
    end

    @testset "5. Matrix factorized_cross and factorized_cross_te" begin
        cl = [1.0 / (l^2) for l in ells]
        mbb = ModifiedBlackbodySED(150.0, 19.6)
        F = eval_chromatic_sed_bands(ν -> sed_weight(mbb, ν, 1.75), bands, chrom_beams)
        @test size(F) == (2, n_ell)

        D = factorized_cross(F, cl)
        @test size(D) == (2, 2, n_ell)

        # Check elementwise formula D[i,j,l] = F[i,l]*F[j,l]*cl[l]
        for i in 1:2, j in 1:2, ℓ in 1:n_ell
            @test isapprox(D[i, j, ℓ], F[i, ℓ] * F[j, ℓ] * cl[ℓ]; rtol=1e-12)
        end

        # Achromatic equivalence
        f_vec = eval_sed_bands(ν -> sed_weight(mbb, ν, 1.75), bands)
        F_achrom = repeat(f_vec, 1, n_ell)
        D_achrom = factorized_cross(F_achrom, cl)
        D_vec = factorized_cross(f_vec, cl)
        @test isapprox(D_achrom, D_vec; rtol=1e-12)

        # TE cross
        D_te = factorized_cross_te(F, 2.0 .* F, cl)
        @test size(D_te) == (2, 2, n_ell)
        for i in 1:2, j in 1:2, ℓ in 1:n_ell
            @test isapprox(D_te[i, j, ℓ], F[i, ℓ] * (2.0 * F[j, ℓ]) * cl[ℓ]; rtol=1e-12)
        end

        # eval_component with chromatic beams
        comp = SkyComponent(mbb, PowerLawShape(3000.0))
        D_comp = eval_component(comp, ells, bands, chrom_beams, 5.0, 1.75; alpha=-0.6)
        @test size(D_comp) == (2, 2, n_ell)
        @test isapprox(D_comp, factorized_cross(F, angular_power(comp.angular, ells; amp=5.0, alpha=-0.6)))

        @test !applicable(factorized_cross, complex.(F), cl)
        @test !applicable(factorized_cross_te, complex.(F), complex.(F), cl)
    end

    @testset "6. Autodiff on matrix factorized_cross" begin
        cl = [1.0 / (l^2) for l in ells]
        F = [1.0 + 0.1 * i + 0.01 * (l/1000.0) for i in 1:2, l in ells]

        # ForwardDiff vs Mooncake on factorized_cross
        W = [(-1.0)^i * (i + 2j + 3l) for i in 1:2, j in 1:2, l in 1:n_ell]
        loss_F = F_mat -> sum(W .* factorized_cross(F_mat, cl))
        loss_cl = cl_vec -> sum(W .* factorized_cross(F, cl_vec))

        g_F_fd = DI.gradient(loss_F, AutoForwardDiff(), F)
        g_F_mc = DI.gradient(loss_F, AutoMooncake(config=nothing), F)
        @test isapprox(g_F_fd, g_F_mc; rtol=1e-6)

        g_cl_fd = DI.gradient(loss_cl, AutoForwardDiff(), cl)
        g_cl_mc = DI.gradient(loss_cl, AutoMooncake(config=nothing), cl)
        @test isapprox(g_cl_fd, g_cl_mc; rtol=1e-6)

        # TE loss
        FE = 2.0 .* F .+ 0.3
        loss_te_FT = FT -> sum(W .* factorized_cross_te(FT, FE, cl))
        loss_te_FE = x -> sum(W .* factorized_cross_te(F, x, cl))
        g_te_fd = DI.gradient(loss_te_FT, AutoForwardDiff(), F)
        g_te_mc = DI.gradient(loss_te_FT, AutoMooncake(config=nothing), F)
        @test isapprox(g_te_fd, g_te_mc; rtol=1e-6)
        @test DI.gradient(loss_te_FE, AutoForwardDiff(), FE) ≈
              DI.gradient(loss_te_FE, AutoMooncake(config=nothing), FE) rtol=1e-6

        # Fixed integer templates must not force integer cotangents.
        int_cl = collect(1:n_ell)
        int_loss = x -> sum(W .* factorized_cross(x, int_cl))
        @test DI.gradient(int_loss, AutoForwardDiff(), F) ≈
              DI.gradient(int_loss, AutoZygote(), F) rtol=1e-10

        F_diag = Diagonal([1.0, 2.0])
        D_diag, pb_diag = ChainRulesCore.rrule(factorized_cross, F_diag, ones(2))
        _, dF_diag, dcl_diag = pb_diag(ones(size(D_diag)))
        @test dF_diag isa Diagonal
        @test Matrix(dF_diag) == Matrix(Diagonal([2.0, 4.0]))
        @test dcl_diag == [1.0, 4.0]

        F_view = @view reshape(collect(1.0:9.0), 3, 3)[1:2, 1:2]
        D_view, pb_view = ChainRulesCore.rrule(factorized_cross, F_view, ones(2))
        _, dF_view, dcl_view = pb_view(ones(size(D_view)))
        @test dF_view == [6.0 18.0; 6.0 18.0]
        @test dcl_view == [9.0, 81.0]

        FT_diag = Diagonal([1.0, 2.0])
        FE_diag = Diagonal([3.0, 4.0])
        D_te_diag, pb_te_diag = ChainRulesCore.rrule(
            factorized_cross_te, FT_diag, FE_diag, ones(2)
        )
        _, dFT_diag, dFE_diag, dcl_te_diag = pb_te_diag(ones(size(D_te_diag)))
        @test dFT_diag isa Diagonal
        @test dFE_diag isa Diagonal
        @test Matrix(dFT_diag) == Matrix(Diagonal([3.0, 4.0]))
        @test Matrix(dFE_diag) == Matrix(Diagonal([1.0, 2.0]))
        @test dcl_te_diag == [3.0, 8.0]
    end

    @testset "7. Type stability (JET)" begin
        cl = [1.0 / (l^2) for l in ells]
        F = [1.0 + 0.1 * i + 0.01 * (l/1000.0) for i in 1:2, l in ells]
        @test_opt factorized_cross(F, cl)
        @test_opt factorized_cross_te(F, F, cl)
    end
end
