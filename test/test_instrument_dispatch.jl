using Test
using CMBForegrounds
using LinearAlgebra
using ForwardDiff
using Mooncake
using Zygote
using DifferentiationInterface
const DI = DifferentiationInterface
using JET

@testset "Phase 5 — Instrumental Operations" begin
    ells = collect(range(100.0, 3000.0, length=50))
    n_ell = length(ells)
    cl = [1.0 / (l^2) for l in ells]

    # ----------------------------------------------------------------- #
    # 1. Calibration                                                    #
    # ----------------------------------------------------------------- #
    @testset "1. Calibration" begin
        c1, c2 = 1.02, 0.98
        # Forward
        @test calibration_factor(c1, c2; convention=:forward) ≈ c1 * c2
        @test apply_calibration(cl, c1, c2; convention=:forward) ≈ (c1 * c2) .* cl
        # Inverse
        @test calibration_factor(c1, c2; convention=:inverse) ≈ 1 / (c1 * c2)
        @test apply_calibration(cl, c1, c2; convention=:inverse) ≈ cl ./ (c1 * c2)

        # 3D tensor calibration with map gains
        gains = [1.02, 0.98, 1.01]
        D = ones(3, 3, n_ell)
        D_cal_fwd = apply_calibration(D, gains; convention=:forward)
        @test size(D_cal_fwd) == (3, 3, n_ell)
        for i in 1:3, j in 1:3
            @test D_cal_fwd[i, j, :] ≈ fill(gains[i] * gains[j], n_ell)
        end

        D_cal_inv = apply_calibration(D, gains; convention=:inverse)
        for i in 1:3, j in 1:3
            @test D_cal_inv[i, j, :] ≈ fill(1 / (gains[i] * gains[j]), n_ell)
        end

        # Independent spectrum gains matrix
        G_mat = [1.01 1.02 1.03; 1.02 1.00 0.99; 1.03 0.99 1.05]
        D_mat_cal = apply_calibration(D, G_mat; convention=:forward)
        for i in 1:3, j in 1:3
            @test D_mat_cal[i, j, :] ≈ fill(G_mat[i, j], n_ell)
        end

        # AD on calibration
        loss_cal = g -> sum(apply_calibration(D, g; convention=:forward))
        g_fd = ForwardDiff.gradient(loss_cal, gains)
        g_mc = DI.gradient(loss_cal, AutoMooncake(config=nothing), gains)
        @test isapprox(g_fd, g_mc; rtol=1e-6)
    end

    # ----------------------------------------------------------------- #
    # 2. Additive templates                                             #
    # ----------------------------------------------------------------- #
    @testset "2. Additive templates" begin
        template = [1.0 / l for l in ells]
        amp = 2.5

        # Single spectrum
        @test additive_template(template, amp) ≈ amp .* template
        @test add_template(cl, template, amp) ≈ cl .+ amp .* template
        # Fixed amplitude amp=1.0 default
        @test additive_template(template) == template
        @test add_template(cl, template) == cl .+ template

        # 3D tensor
        amp_mat = [1.0 0.5; 0.5 2.0]
        D2 = ones(2, 2, n_ell)
        D_add = add_template(D2, template, amp_mat)
        @test size(D_add) == (2, 2, n_ell)
        for i in 1:2, j in 1:2
            @test D_add[i, j, :] ≈ 1.0 .+ amp_mat[i, j] .* template
        end

        # AD on template amplitude
        loss_amp = a -> sum(add_template(cl, template, a[1]))
        g_fd = DI.gradient(loss_amp, AutoForwardDiff(), [amp])
        g_mc = DI.gradient(loss_amp, AutoMooncake(config=nothing), [amp])
        @test isapprox(g_fd, g_mc; rtol=1e-6)
    end

    # ----------------------------------------------------------------- #
    # 3. Polarization leakage                                           #
    # ----------------------------------------------------------------- #
    @testset "3. Polarization leakage" begin
        C_TT = cl
        C_TE = 0.3 .* cl
        gamma1 = 0.05
        gamma2 = 0.03

        # TE leakage: gamma_j * C_TT
        dC_TE = te_leakage(C_TT, gamma2)
        @test dC_TE ≈ gamma2 .* C_TT

        # ET leakage: gamma_i * C_TT
        dC_ET = et_leakage(C_TT, gamma1)
        @test dC_ET ≈ gamma1 .* C_TT

        # EE leakage cross
        dC_EE_cross = ee_leakage(C_TT, C_TE, C_TE, gamma1, gamma2)
        expected_ee = gamma1 .* C_TE .+ gamma2 .* C_TE .+ (gamma1 * gamma2) .* C_TT
        @test dC_EE_cross ≈ expected_ee

        # EE leakage auto
        dC_EE_auto = ee_leakage(C_TT, C_TE, gamma1)
        @test dC_EE_auto ≈ 2 * gamma1 .* C_TE .+ (gamma1^2) .* C_TT

        # Multipole-dependent leakage curve gamma(ell)
        gamma_curve = [0.02 * (l / 1000.0) for l in ells]
        dC_TE_curve = te_leakage(C_TT, gamma_curve)
        @test dC_TE_curve ≈ gamma_curve .* C_TT

        # 3D tensor leakage
        D_TE = ones(2, 2, n_ell)
        D_TT = 2.0 .* ones(2, 2, n_ell)
        gammas = [0.04, 0.06]
        D_TE_obs = apply_te_leakage(D_TE, D_TT, gammas)
        for i in 1:2, j in 1:2
            @test D_TE_obs[i, j, :] ≈ fill(1.0 + gammas[j] * 2.0, n_ell)
        end

        # AD on leakage
        loss_leak = g -> sum(ee_leakage(C_TT, C_TE, g[1]))
        g_fd = DI.gradient(loss_leak, AutoForwardDiff(), [0.05])
        g_mc = DI.gradient(loss_leak, AutoMooncake(config=nothing), [0.05])
        @test isapprox(g_fd, g_mc; rtol=1e-6)
    end

    # ----------------------------------------------------------------- #
    # 4. Beam eigenmodes                                                #
    # ----------------------------------------------------------------- #
    @testset "4. Beam eigenmodes" begin
        n_modes = 3
        modes = [0.01 * (l / 1000.0)^k for l in ells, k in 1:n_modes]
        coeffs = [0.5, -0.2, 0.1]

        # Non-linearized: cl * (1 + sum beta_k phi_k)^2
        perturbed = beam_eigenmode_response(cl, modes, coeffs; linearized=false)
        delta_b = modes * coeffs
        @test perturbed ≈ cl .* (1 .+ delta_b).^2

        # Linearized: 2 * cl * delta_b
        perturbed_lin = beam_eigenmode_response(cl, modes, coeffs; linearized=true)
        @test perturbed_lin ≈ 2 .* cl .* delta_b

        # Cross beam eigenmodes
        coeffs_j = [0.1, 0.2, -0.1]
        cross_pert = beam_eigenmode_cross(cl, modes, coeffs, modes, coeffs_j; linearized=false)
        delta_bj = modes * coeffs_j
        @test cross_pert ≈ cl .* (1 .+ delta_b) .* (1 .+ delta_bj)

        # AD on beam mode coefficients
        loss_beam = c -> sum(beam_eigenmode_response(cl, modes, c; linearized=false))
        g_fd = ForwardDiff.gradient(loss_beam, coeffs)
        g_mc = DI.gradient(loss_beam, AutoMooncake(config=nothing), coeffs)
        @test isapprox(g_fd, g_mc; rtol=1e-6)
    end

    # ----------------------------------------------------------------- #
    # 5. SSL and Aberration                                             #
    # ----------------------------------------------------------------- #
    @testset "5. SSL and Aberration" begin
        kappa = 0.01
        ab_coeff = 0.001

        # Check that ssl_response returns Delta D_ell (correction)
        delta_ssl = ssl_response(ells, kappa, cl)
        @test delta_ssl ≈ -kappa .* (ells .* ells .* (ells .+ 1) ./ (2π) .* dCl_dell_from_Dl(ells, cl) .+ 2 .* cl)

        # apply_ssl returns corrected spectrum cl + Delta D_ell
        cl_ssl = apply_ssl(ells, kappa, cl)
        @test cl_ssl ≈ cl .+ delta_ssl

        # aberration_response returns Delta D_ell (correction)
        delta_ab = aberration_response(ells, ab_coeff, cl)
        @test delta_ab ≈ -ab_coeff .* dCl_dell_from_Dl(ells, cl) .* (ells .* ells .* (ells .+ 1) ./ (2π))

        # apply_aberration returns cl + Delta D_ell
        cl_ab = apply_aberration(ells, ab_coeff, cl)
        @test cl_ab ≈ cl .+ delta_ab

        # Zero coefficient limit: identity
        @test apply_ssl(ells, 0.0, cl) ≈ cl
        @test apply_aberration(ells, 0.0, cl) ≈ cl
    end

    # ----------------------------------------------------------------- #
    # 6. Type stability (JET)                                           #
    # ----------------------------------------------------------------- #
    @testset "6. Type stability (JET)" begin
        @test_opt calibration_factor(1.0, 1.0)
        @test_opt apply_calibration(cl, 1.02, 0.98)
        @test_opt additive_template(cl, 2.0)
        @test_opt add_template(cl, cl, 2.0)
        @test_opt te_leakage(cl, 0.05)
        @test_opt ee_leakage(cl, cl, 0.05)
    end
end
