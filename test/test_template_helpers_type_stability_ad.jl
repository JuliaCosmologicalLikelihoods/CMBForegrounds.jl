"""
Type-stability (JET) + autodiff (DifferentiationInterface) tests for
the three generic ℓ-template helpers:

  eval_template(T, ell, ell_0; amp)
  eval_template_tilt(T, ell, ell_0, alpha; amp)
  eval_powerlaw(ell, ell_0, alpha; amp)

These are the building blocks for kSZ/tSZ/CIB/dust ℓ-shapes.

The free parameters we differentiate w.r.t.:
  - `amp`   (linear scaling — straightforward)
  - `alpha` (power-law exponent — non-trivial for Dual numbers because
             ell is an integer vector; `(ell/ell_0)^alpha` must work for
             Dual `alpha`)
"""

using JET
using ADTypes
import DifferentiationInterface as DI
using ForwardDiff
using Zygote
using Mooncake

@testset "eval_template — type stability + AD" begin

    T    = Float64.(1:200)        # fake template
    ell  = collect(2:101)         # 100 ℓ values (Int)
    ell_0 = 50                    # pivot (Int)
    amp   = 3.5

    w = randn(MersenneTwister(1), length(ell))

    # Scalar loss ℓ(amp) = Σ w_i * output_i
    f_amp = x -> sum(w .* CMBForegrounds.eval_template(T, ell, ell_0; amp=x[1]))
    x0_amp = [amp]

    @testset "JET @test_opt — Float64" begin
        JET.@test_opt CMBForegrounds.eval_template(T, ell, ell_0; amp=amp)
    end

    @testset "JET @test_opt — Dual amp" begin
        amp_d = ForwardDiff.Dual(amp, 1.0)
        JET.@test_opt target_modules=(CMBForegrounds,) CMBForegrounds.eval_template(T, ell, ell_0; amp=amp_d)
    end

    @testset "DI gradient wrt amp — ForwardDiff" begin
        g = DI.gradient(f_amp, AutoForwardDiff(), x0_amp)
        @test all(isfinite, g)
        # Analytical: dℓ/d_amp = Σ w_i T[ell_i+1] / T[ell_0+1]
        norm = T[ell_0 + 1]
        @test g[1] ≈ sum(w .* T[ell .+ 1] ./ norm) rtol = 1e-10
    end

    @testset "DI gradient wrt amp — Zygote" begin
        g = DI.gradient(f_amp, AutoZygote(), x0_amp)
        @test all(isfinite, g)
        norm = T[ell_0 + 1]
        @test g[1] ≈ sum(w .* T[ell .+ 1] ./ norm) rtol = 1e-10
    end

    @testset "DI gradient wrt amp — Mooncake" begin
        g = DI.gradient(f_amp, AutoMooncake(; config=nothing), x0_amp)
        @test all(isfinite, g)
        norm = T[ell_0 + 1]
        @test g[1] ≈ sum(w .* T[ell .+ 1] ./ norm) rtol = 1e-10
    end
end

@testset "eval_template_tilt — type stability + AD" begin

    T     = Float64.(1:200)
    ell   = collect(2:101)
    ell_0 = 50
    amp   = 3.5
    alpha = -0.53          # realistic tSZ tilt (ACT DR6)

    w = randn(MersenneTwister(2), length(ell))

    # Loss over both amp and alpha packed into a length-2 vector
    f = x -> sum(w .* CMBForegrounds.eval_template_tilt(T, ell, ell_0, x[2]; amp=x[1]))
    x0 = [amp, alpha]

    @testset "JET @test_opt — Float64" begin
        JET.@test_opt CMBForegrounds.eval_template_tilt(T, ell, ell_0, alpha; amp=amp)
    end

    @testset "JET @test_opt — Dual alpha" begin
        alpha_d = ForwardDiff.Dual(alpha, 1.0)
        JET.@test_opt target_modules=(CMBForegrounds,) CMBForegrounds.eval_template_tilt(T, ell, ell_0, alpha_d; amp=amp)
    end

    @testset "DI gradient wrt [amp, alpha] — ForwardDiff" begin
        g = DI.gradient(f, AutoForwardDiff(), x0)
        @test length(g) == 2
        @test all(isfinite, g)
    end

    @testset "DI gradient wrt [amp, alpha] — Zygote" begin
        g = DI.gradient(f, AutoZygote(), x0)
        @test length(g) == 2
        @test all(isfinite, g)
    end

    @testset "DI gradient wrt [amp, alpha] — Mooncake" begin
        g = DI.gradient(f, AutoMooncake(; config=nothing), x0)
        @test length(g) == 2
        @test all(isfinite, g)
    end

    @testset "Cross-backend agreement" begin
        gFD = DI.gradient(f, AutoForwardDiff(),                x0)
        gZG = DI.gradient(f, AutoZygote(),                     x0)
        gMC = DI.gradient(f, AutoMooncake(; config=nothing),   x0)
        @test gFD ≈ gZG rtol = 1e-6
        @test gFD ≈ gMC rtol = 1e-6
    end

    @testset "Analytical gradient wrt amp" begin
        # d/d_amp = Σ w_i T[ell_i+1]/T[ell_0+1] * (ell_i/ell_0)^alpha
        norm  = T[ell_0 + 1]
        d_amp = sum(w .* T[ell .+ 1] ./ norm .* (ell ./ ell_0) .^ alpha)
        g     = DI.gradient(f, AutoForwardDiff(), x0)
        @test g[1] ≈ d_amp rtol = 1e-10
    end
end

@testset "eval_powerlaw — type stability + AD" begin

    ell   = Float64.(collect(2:101))    # Float for ℓ(ℓ+1) style inputs
    ell_0 = 3000.0
    amp   = 6.0
    alpha = -2.76          # realistic radio spectral index (ACT DR6)

    w = randn(MersenneTwister(3), length(ell))

    f = x -> sum(w .* CMBForegrounds.eval_powerlaw(ell, ell_0, x[2]; amp=x[1]))
    x0 = [amp, alpha]

    @testset "JET @test_opt — Float64" begin
        JET.@test_opt CMBForegrounds.eval_powerlaw(ell, ell_0, alpha; amp=amp)
    end

    @testset "JET @test_opt — Dual alpha" begin
        alpha_d = ForwardDiff.Dual(alpha, 1.0)
        JET.@test_opt target_modules=(CMBForegrounds,) CMBForegrounds.eval_powerlaw(ell, ell_0, alpha_d; amp=amp)
    end

    @testset "DI gradient — ForwardDiff" begin
        g = DI.gradient(f, AutoForwardDiff(), x0)
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
        gFD = DI.gradient(f, AutoForwardDiff(),                x0)
        gZG = DI.gradient(f, AutoZygote(),                     x0)
        gMC = DI.gradient(f, AutoMooncake(; config=nothing),   x0)
        @test gFD ≈ gZG rtol = 1e-6
        @test gFD ≈ gMC rtol = 1e-6
    end
end
