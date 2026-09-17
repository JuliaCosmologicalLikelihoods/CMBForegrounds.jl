using Test
using CMBForegrounds
using LinearAlgebra
using ForwardDiff
using Mooncake
using Zygote
using DifferentiationInterface
using ADTypes
using ChainRulesCore
using FiniteDifferences
const DI = DifferentiationInterface
using JET

@testset "Instrumental operations" begin
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
        g_fd = DI.gradient(loss_cal, AutoForwardDiff(), gains)
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

        @test_throws DimensionMismatch add_template(cl, template[1:end-1], amp)
        @test_throws DimensionMismatch add_template(D2, template[1:end-1], amp_mat)
        @test_throws DimensionMismatch add_template(D2, template, ones(1, 1))
        @test_throws DimensionMismatch add_template(ones(2, 1, n_ell), template,
                                                     ones(2, 1))
    end

    # ----------------------------------------------------------------- #
    # 3. Polarization leakage                                           #
    # ----------------------------------------------------------------- #
    @testset "3. Polarization leakage" begin
        C_TT = cl
        C_TE = 0.3 .* cl
        C_ET = 0.7 .* cl
        gamma1 = 0.05
        gamma2 = 0.03

        # TE leakage: gamma_j * C_TT
        dC_TE = te_leakage(C_TT, gamma2)
        @test dC_TE ≈ gamma2 .* C_TT

        # ET leakage: gamma_i * C_TT
        dC_ET = et_leakage(C_TT, gamma1)
        @test dC_ET ≈ gamma1 .* C_TT

        # EE leakage cross
        dC_EE_cross = ee_leakage(C_TT, C_TE, C_ET, gamma1, gamma2)
        expected_ee = gamma1 .* C_TE .+ gamma2 .* C_ET .+ (gamma1 * gamma2) .* C_TT
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

        # Preserve ordered TE/ET legs in the EE map-response algebra.
        D_TE_ordered = [10i + j + 0.01l for i in 1:2, j in 1:2, l in 1:n_ell]
        D_EE = zeros(2, 2, n_ell)
        D_EE_obs = apply_ee_leakage(D_EE, D_TE_ordered, D_TT, gammas)
        for i in 1:2, j in 1:2
            expected = @. gammas[i] * D_TE_ordered[i, j, :] +
                           gammas[j] * D_TE_ordered[j, i, :] +
                           gammas[i] * gammas[j] * D_TT[i, j, :]
            @test D_EE_obs[i, j, :] ≈ expected
        end

        # AD on leakage
        loss_leak = g -> sum(ee_leakage(C_TT, C_TE, g[1]))
        g_fd = DI.gradient(loss_leak, AutoForwardDiff(), [0.05])
        g_mc = DI.gradient(loss_leak, AutoMooncake(config=nothing), [0.05])
        @test isapprox(g_fd, g_mc; rtol=1e-6)

        tensor_loss = g -> sum(apply_ee_leakage(D_EE, D_TE_ordered, D_TT, g))
        tensor_fd = DI.gradient(tensor_loss, AutoForwardDiff(), gammas)
        @test tensor_fd ≈ DI.gradient(tensor_loss, AutoMooncake(config=nothing), gammas) rtol=1e-6
        @test tensor_fd ≈ DI.gradient(tensor_loss, AutoZygote(), gammas) rtol=1e-8

        @test_throws DimensionMismatch te_leakage(C_TT, gamma_curve[1:end-1])
        @test_throws DimensionMismatch et_leakage(C_TT, gamma_curve[1:end-1])
        @test_throws DimensionMismatch ee_leakage(
            C_TT, C_TE[1:end-1], C_ET, gamma1, gamma2
        )
        @test_throws DimensionMismatch ee_leakage(
            C_TT, C_TE, gamma_curve[1:end-1]
        )
        @test_throws DimensionMismatch ee_leakage(
            C_TT, C_TE, C_ET, gamma_curve[1:end-1], gamma_curve
        )
        @test_throws DimensionMismatch ee_leakage(
            C_TT, C_TE, C_ET, gamma_curve, gamma_curve[1:end-1]
        )
        @test_throws DimensionMismatch apply_te_leakage(
            zeros(2, 2, 1), ones(2, 2, 3), gammas
        )
        @test_throws DimensionMismatch apply_te_leakage(
            zeros(2, 1, 1), zeros(2, 1, 1), gammas
        )
        @test_throws DimensionMismatch apply_te_leakage(
            zeros(2, 2, 1), zeros(2, 2, 1), [0.1]
        )
        @test_throws DimensionMismatch apply_ee_leakage(
            zeros(2, 2, 1), zeros(2, 2, 1), ones(2, 2, 3), gammas
        )
        @test_throws DimensionMismatch apply_ee_leakage(
            zeros(2, 1, 1), zeros(2, 1, 1), zeros(2, 1, 1), gammas
        )
        @test_throws DimensionMismatch apply_ee_leakage(
            zeros(2, 2, 1), zeros(2, 2, 1), zeros(2, 2, 1), [0.1]
        )
        @test size(apply_te_leakage(
            zeros(1, 1, 1), ones(1, 1, 1), [0.1]
        )) == (1, 1, 1)
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

        # Linearized corrected spectrum: cl * (1 + 2delta_b)
        perturbed_lin = beam_eigenmode_response(cl, modes, coeffs; linearized=true)
        @test perturbed_lin ≈ cl .* (1 .+ 2 .* delta_b)
        @test beam_eigenmode_response(cl, modes, zeros(n_modes); linearized=true) == cl

        # Cross beam eigenmodes
        coeffs_j = [0.1, 0.2, -0.1]
        cross_pert = beam_eigenmode_cross(cl, modes, coeffs, modes, coeffs_j; linearized=false)
        delta_bj = modes * coeffs_j
        @test cross_pert ≈ cl .* (1 .+ delta_b) .* (1 .+ delta_bj)

        # AD on beam mode coefficients
        loss_beam = c -> sum(beam_eigenmode_response(cl, modes, c; linearized=false))
        g_fd = DI.gradient(loss_beam, AutoForwardDiff(), coeffs)
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

    @testset "7. Fixed window convolution" begin
        window = rand(7, 3)
        spectrum = rand(7)
        projection_bar = randn(3)
        projection, pullback = ChainRulesCore.rrule(window_convolution, window, spectrum)
        function_bar, window_bar, spectrum_bar = pullback(projection_bar)
        @test projection ≈ transpose(window) * spectrum
        @test function_bar isa ChainRulesCore.NoTangent
        @test window_bar isa ChainRulesCore.NoTangent
        @test spectrum_bar ≈ window * projection_bar

        weights = randn(3)
        objective(x) = dot(weights, window_convolution(window, x))
        gradient_forward = DI.gradient(objective, AutoForwardDiff(), spectrum)
        gradient_mooncake = DI.gradient(objective, AutoMooncake(config=nothing), spectrum)
        gradient_finite = DI.gradient(
            objective,
            AutoFiniteDifferences(; fdm=FiniteDifferences.central_fdm(5, 1)), spectrum,
        )
        @test gradient_mooncake ≈ gradient_forward rtol=1e-12
        @test gradient_mooncake ≈ gradient_finite rtol=1e-8 atol=1e-10
        @test_throws DimensionMismatch window_convolution(window, rand(6))
    end
end
