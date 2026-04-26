"""
Type-stability (JET) + autodiff (DifferentiationInterface) tests for
`ksz_template_scaled(template, AkSZ)`.

Function under test: `D_ℓ = template .* AkSZ`.

Coverage:
1. JET `@test_opt` — no inference fallbacks on Float64 inputs.
2. JET `@test_opt` — same on a ForwardDiff Dual amplitude (scoped to
   `CMBForegrounds` so ForwardDiff internals aren't analysed).
3. DI gradient agreement across ForwardDiff / Zygote / Mooncake.
"""

using JET
using ADTypes
import DifferentiationInterface as DI
using ForwardDiff
using Zygote
using Mooncake

@testset "ksz_template_scaled — type stability + AD" begin

    template = collect(1.0:10.0)
    weight   = randn(MersenneTwister(0), 10)
    AkSZ     = 1.7

    # Scalar loss for gradient testing: ℓ(A) = Σ_i w_i (template_i · A).
    f_vec = x -> sum(weight .* CMBForegrounds.ksz_template_scaled(template, x[1]))
    x0    = [AkSZ]

    @testset "JET @test_opt — Float64" begin
        JET.@test_opt CMBForegrounds.ksz_template_scaled(template, AkSZ)
    end

    @testset "JET @test_opt — ForwardDiff Dual" begin
        # Build a Dual via the no-tag two-arg constructor.
        AkSZ_d = ForwardDiff.Dual(AkSZ, 1.0)
        JET.@test_opt target_modules = (CMBForegrounds,) CMBForegrounds.ksz_template_scaled(template, AkSZ_d)
    end

    @testset "DI gradient — ForwardDiff" begin
        backend = AutoForwardDiff()
        prep    = DI.prepare_gradient(f_vec, backend, x0)
        g       = DI.gradient(f_vec, prep, backend, x0)
        @test length(g) == 1
        @test all(isfinite, g)
        @test g[1] ≈ sum(weight .* template) rtol = 1e-10
    end

    @testset "DI gradient — Zygote" begin
        backend = AutoZygote()
        prep    = DI.prepare_gradient(f_vec, backend, x0)
        g       = DI.gradient(f_vec, prep, backend, x0)
        @test length(g) == 1
        @test all(isfinite, g)
        @test g[1] ≈ sum(weight .* template) rtol = 1e-10
    end

    @testset "DI gradient — Mooncake" begin
        backend = AutoMooncake(; config = nothing)
        prep    = DI.prepare_gradient(f_vec, backend, x0)
        g       = DI.gradient(f_vec, prep, backend, x0)
        @test length(g) == 1
        @test all(isfinite, g)
        @test g[1] ≈ sum(weight .* template) rtol = 1e-10
    end

    @testset "Cross-backend agreement" begin
        gFD = DI.gradient(f_vec, AutoForwardDiff(),                x0)
        gZG = DI.gradient(f_vec, AutoZygote(),                     x0)
        gMC = DI.gradient(f_vec, AutoMooncake(; config = nothing), x0)
        @test gFD ≈ gZG rtol = 1e-10
        @test gFD ≈ gMC rtol = 1e-10
    end
end
