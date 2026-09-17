using ADTypes
using ChainRulesCore
using DifferentiationInterface
using FiniteDifferences
using ForwardDiff
using LinearAlgebra
using Mooncake

@testset "Chromatic fused foreground assemblers" begin
    rng = Xoshiro(0xC4A0)
    n_freq, n_ell = 3, 4
    amplitudes = rand(rng, 7)
    matrices = [rand(rng, n_freq, n_ell) for _ in 1:8]
    spectra = [rand(rng, n_ell) for _ in 1:7]
    weights_TT = randn(rng, n_freq, n_freq, n_ell)
    weights_TE = randn(rng, n_freq, n_freq, n_ell)
    weights_EE = randn(rng, n_freq, n_freq, n_ell)

    a_p, a_gtt, a_s, a_pste, a_gte, a_psee, a_gee = amplitudes
    f_ksz, f_cibp, f_dust_T, f_radio_T, f_tsz, f_cibc,
        f_radio_P, f_dust_P = matrices
    cl_ksz, cl_cibp, cl_dust_T, cl_radio, cl_tsz, cl_cibc, cl_szxcib = spectra

    TT = assemble_TT(a_p, a_gtt, a_s,
                     f_ksz, f_cibp, f_dust_T, f_radio_T, f_tsz, f_cibc,
                     cl_ksz, cl_cibp, cl_dust_T, cl_radio,
                     cl_tsz, cl_cibc, cl_szxcib)
    TT_reference = factorized_cross(f_ksz, cl_ksz) .+
                   factorized_cross(f_tsz, cl_tsz) .+
                   factorized_cross(f_cibc, cl_cibc) .+
                   factorized_cross_te(f_tsz, f_cibc, cl_szxcib) .+
                   factorized_cross_te(f_cibc, f_tsz, cl_szxcib) .+
                   a_p .* factorized_cross(f_cibp, cl_cibp) .+
                   a_gtt .* factorized_cross(f_dust_T, cl_dust_T) .+
                   a_s .* factorized_cross(f_radio_T, cl_radio)
    @test TT ≈ TT_reference rtol=1e-12

    TE = assemble_TE(a_pste, a_gte, f_radio_T, f_radio_P,
                     f_dust_T, f_dust_P, cl_radio, cl_dust_T)
    @test TE ≈ a_pste .* factorized_cross_te(f_radio_T, f_radio_P, cl_radio) .+
               a_gte .* factorized_cross_te(f_dust_T, f_dust_P, cl_dust_T) rtol=1e-12

    EE = assemble_EE(a_psee, a_gee, f_radio_P, f_dust_P, cl_radio, cl_dust_T)
    @test EE ≈ a_psee .* factorized_cross(f_radio_P, cl_radio) .+
               a_gee .* factorized_cross(f_dust_P, cl_dust_T) rtol=1e-12

    matrix_length = n_freq * n_ell
    function objective(v)
        a = v[1:7]
        offset = 8
        f = ntuple(8) do _
            result = reshape(v[offset:offset + matrix_length - 1], n_freq, n_ell)
            offset += matrix_length
            result
        end
        cl = ntuple(7) do _
            result = v[offset:offset + n_ell - 1]
            offset += n_ell
            result
        end
        tt = assemble_TT(a[1], a[2], a[3], f[1], f[2], f[3], f[4], f[5], f[6],
                         cl[1], cl[2], cl[3], cl[4], cl[5], cl[6], cl[7])
        te = assemble_TE(a[4], a[5], f[4], f[7], f[3], f[8], cl[4], cl[3])
        ee = assemble_EE(a[6], a[7], f[7], f[8], cl[4], cl[3])
        return dot(weights_TT, tt) + dot(weights_TE, te) + dot(weights_EE, ee)
    end
    input = vcat(amplitudes, vec.(matrices)..., spectra...)
    gradient_forward = DifferentiationInterface.gradient(objective, AutoForwardDiff(), input)
    gradient_mooncake = DifferentiationInterface.gradient(
        objective, AutoMooncake(; config=nothing), input,
    )
    gradient_finite = DifferentiationInterface.gradient(
        objective, AutoFiniteDifferences(; fdm=FiniteDifferences.central_fdm(5, 1)), input,
    )
    @test all(isfinite, gradient_mooncake)
    @test gradient_mooncake ≈ gradient_forward rtol=1e-10
    @test gradient_mooncake ≈ gradient_finite rtol=1e-5 atol=1e-7
end

@testset "Chromatic ratio fixed-beam pullback" begin
    rng = Xoshiro(0xC4A1)
    beam = rand(rng, 7, 4)
    weights = rand(rng, 4)
    sed_values = rand(rng, 4)
    denominator = rand(rng, 7) .+ 1
    output̄ = randn(rng, 7)

    output, pullback = ChainRulesCore.rrule(
        CMBForegrounds._fixed_chromatic_ratio, beam, weights, sed_values, denominator,
    )
    function̄, beam̄, weights̄, sed_values̄, denominator̄ = pullback(output̄)
    scaled_output̄ = output̄ ./ denominator
    weighted_sed̄ = transpose(beam) * scaled_output̄
    @test function̄ isa ChainRulesCore.NoTangent
    @test beam̄ isa ChainRulesCore.NoTangent
    @test weights̄ ≈ sed_values .* weighted_sed̄
    @test sed_values̄ ≈ weights .* weighted_sed̄
    @test denominator̄ ≈ -scaled_output̄ .* output

    n_weights = length(weights)
    function objective(v)
        w = v[1:n_weights]
        s = v[n_weights + 1:2n_weights]
        d = v[2n_weights + 1:end]
        return dot(output̄, CMBForegrounds._fixed_chromatic_ratio(beam, w, s, d))
    end
    input = vcat(weights, sed_values, denominator)
    gradient_forward = DifferentiationInterface.gradient(objective, AutoForwardDiff(), input)
    gradient_mooncake = DifferentiationInterface.gradient(
        objective, AutoMooncake(; config=nothing), input,
    )
    gradient_finite = DifferentiationInterface.gradient(
        objective, AutoFiniteDifferences(; fdm=FiniteDifferences.central_fdm(5, 1)), input,
    )
    @test gradient_mooncake ≈ gradient_forward rtol=1e-12
    @test gradient_mooncake ≈ gradient_finite rtol=1e-6 atol=1e-8

    product, product_pullback = ChainRulesCore.rrule(
        CMBForegrounds._fixed_beam_product, beam, weights,
    )
    product_function̄, product_beam̄, product_weights̄ = product_pullback(output̄)
    @test product ≈ beam * weights
    @test product_function̄ isa ChainRulesCore.NoTangent
    @test product_beam̄ isa ChainRulesCore.NoTangent
    @test product_weights̄ ≈ transpose(beam) * output̄

    product_objective(x) = dot(output̄, CMBForegrounds._fixed_beam_product(beam, x))
    product_forward = DifferentiationInterface.gradient(
        product_objective, AutoForwardDiff(), weights,
    )
    product_mooncake = DifferentiationInterface.gradient(
        product_objective, AutoMooncake(; config=nothing), weights,
    )
    product_finite = DifferentiationInterface.gradient(
        product_objective,
        AutoFiniteDifferences(; fdm=FiniteDifferences.central_fdm(5, 1)), weights,
    )
    @test product_mooncake ≈ product_forward rtol=1e-12
    @test product_mooncake ≈ product_finite rtol=1e-7 atol=1e-9
end
