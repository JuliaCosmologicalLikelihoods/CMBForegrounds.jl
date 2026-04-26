"""
Type-stability (JET) + autodiff (DifferentiationInterface) tests for
the ACT-compatible SED functions:

  tsz_sed(nu, nu_0)            — tSZ non-relativistic SED
  mbb_sed(nu, nu_0, beta, temp) — modified blackbody (CIB, dust)
  radio_sed(nu, nu_0, beta)    — radio power-law
  constant_sed(nu)             — kSZ (blackbody, frequency-independent)

Free parameters differentiated wrt:
  tsz_sed:   nu_0 is fixed; nu enters via freq integrals — scalar test only
  mbb_sed:   beta (spectral index), temp (dust temperature)
  radio_sed: beta (spectral index)
"""

using JET
using ADTypes
import DifferentiationInterface as DI
using ForwardDiff
using Zygote
using Mooncake

# Representative ACT DR6 frequencies and reference
const NU_TEST  = [90.0, 150.0, 220.0]   # GHz
const NU_0     = 150.0                   # GHz
const BETA_P   = 2.07                    # CIB Poisson spectral index
const T_DUST   = 9.6                     # K (CIB dust temp)
const BETA_S   = -2.76                   # radio spectral index

# ------------------------------------------------------------------ #
# tsz_sed                                                              #
# ------------------------------------------------------------------ #

@testset "tsz_sed — type stability + AD" begin

    @testset "JET @test_opt — scalar Float64" begin
        JET.@test_opt CMBForegrounds.tsz_sed(150.0, NU_0)
    end

    @testset "JET @test_opt — vector Float64" begin
        JET.@test_opt CMBForegrounds.tsz_sed(NU_TEST, NU_0)
    end

    @testset "JET @test_opt — Dual nu" begin
        nu_d = ForwardDiff.Dual(150.0, 1.0)
        JET.@test_opt target_modules=(CMBForegrounds,) CMBForegrounds.tsz_sed(nu_d, NU_0)
    end

    @testset "Normalization: tsz_sed(nu_0, nu_0) == 1" begin
        @test CMBForegrounds.tsz_sed(NU_0, NU_0) ≈ 1.0
    end

    # Differentiate scalar loss Σ_i tsz_sed(nu_i, nu_0) wrt nu_0 packed in a vector
    # (nu_0 is a nuisance we probe for AD correctness)
    f_tsz = x -> sum(CMBForegrounds.tsz_sed.(NU_TEST, x[1]))
    x0 = [NU_0]

    @testset "DI gradient wrt nu_0 — ForwardDiff" begin
        g = DI.gradient(f_tsz, AutoForwardDiff(), x0)
        @test all(isfinite, g)
    end

    @testset "DI gradient wrt nu_0 — Zygote" begin
        g = DI.gradient(f_tsz, AutoZygote(), x0)
        @test all(isfinite, g)
    end

    @testset "DI gradient wrt nu_0 — Mooncake" begin
        g = DI.gradient(f_tsz, AutoMooncake(; config=nothing), x0)
        @test all(isfinite, g)
    end

    @testset "Cross-backend agreement" begin
        gFD = DI.gradient(f_tsz, AutoForwardDiff(),              x0)
        gZG = DI.gradient(f_tsz, AutoZygote(),                   x0)
        gMC = DI.gradient(f_tsz, AutoMooncake(; config=nothing), x0)
        @test gFD ≈ gZG rtol = 1e-8
        @test gFD ≈ gMC rtol = 1e-8
    end
end

# ------------------------------------------------------------------ #
# mbb_sed                                                              #
# ------------------------------------------------------------------ #

@testset "mbb_sed — type stability + AD" begin

    @testset "JET @test_opt — scalar Float64" begin
        JET.@test_opt CMBForegrounds.mbb_sed(150.0, NU_0, BETA_P, T_DUST)
    end

    @testset "JET @test_opt — Dual beta" begin
        beta_d = ForwardDiff.Dual(BETA_P, 1.0)
        JET.@test_opt target_modules=(CMBForegrounds,) CMBForegrounds.mbb_sed(150.0, NU_0, beta_d, T_DUST)
    end

    @testset "JET @test_opt — Dual temp" begin
        temp_d = ForwardDiff.Dual(T_DUST, 1.0)
        JET.@test_opt target_modules=(CMBForegrounds,) CMBForegrounds.mbb_sed(150.0, NU_0, BETA_P, temp_d)
    end

    @testset "Normalization: mbb_sed(nu_0, nu_0, ...) == 1" begin
        @test CMBForegrounds.mbb_sed(NU_0, NU_0, BETA_P, T_DUST) ≈ 1.0
    end

    # Loss over frequencies: Σ_i mbb_sed(nu_i, nu_0, beta, temp)
    w = randn(MersenneTwister(10), length(NU_TEST))
    f_mbb = x -> sum(w .* CMBForegrounds.mbb_sed.(NU_TEST, NU_0, x[1], x[2]))
    x0 = [BETA_P, T_DUST]

    @testset "DI gradient wrt [beta, temp] — ForwardDiff" begin
        g = DI.gradient(f_mbb, AutoForwardDiff(), x0)
        @test length(g) == 2
        @test all(isfinite, g)
    end

    @testset "DI gradient wrt [beta, temp] — Zygote" begin
        g = DI.gradient(f_mbb, AutoZygote(), x0)
        @test length(g) == 2
        @test all(isfinite, g)
    end

    @testset "DI gradient wrt [beta, temp] — Mooncake" begin
        g = DI.gradient(f_mbb, AutoMooncake(; config=nothing), x0)
        @test length(g) == 2
        @test all(isfinite, g)
    end

    @testset "Cross-backend agreement" begin
        gFD = DI.gradient(f_mbb, AutoForwardDiff(),              x0)
        gZG = DI.gradient(f_mbb, AutoZygote(),                   x0)
        gMC = DI.gradient(f_mbb, AutoMooncake(; config=nothing), x0)
        @test gFD ≈ gZG rtol = 1e-8
        @test gFD ≈ gMC rtol = 1e-8
    end
end

# ------------------------------------------------------------------ #
# radio_sed                                                            #
# ------------------------------------------------------------------ #

@testset "radio_sed — type stability + AD" begin

    @testset "JET @test_opt — scalar Float64" begin
        JET.@test_opt CMBForegrounds.radio_sed(150.0, NU_0, BETA_S)
    end

    @testset "JET @test_opt — Dual beta" begin
        beta_d = ForwardDiff.Dual(BETA_S, 1.0)
        JET.@test_opt target_modules=(CMBForegrounds,) CMBForegrounds.radio_sed(150.0, NU_0, beta_d)
    end

    @testset "Normalization: radio_sed(nu_0, nu_0, beta) == 1" begin
        @test CMBForegrounds.radio_sed(NU_0, NU_0, BETA_S) ≈ 1.0
    end

    w = randn(MersenneTwister(11), length(NU_TEST))
    f_radio = x -> sum(w .* CMBForegrounds.radio_sed.(NU_TEST, NU_0, x[1]))
    x0 = [BETA_S]

    @testset "DI gradient wrt beta — ForwardDiff" begin
        g = DI.gradient(f_radio, AutoForwardDiff(), x0)
        @test all(isfinite, g)
    end

    @testset "DI gradient wrt beta — Zygote" begin
        g = DI.gradient(f_radio, AutoZygote(), x0)
        @test all(isfinite, g)
    end

    @testset "DI gradient wrt beta — Mooncake" begin
        g = DI.gradient(f_radio, AutoMooncake(; config=nothing), x0)
        @test all(isfinite, g)
    end

    @testset "Cross-backend agreement" begin
        gFD = DI.gradient(f_radio, AutoForwardDiff(),              x0)
        gZG = DI.gradient(f_radio, AutoZygote(),                   x0)
        gMC = DI.gradient(f_radio, AutoMooncake(; config=nothing), x0)
        @test gFD ≈ gZG rtol = 1e-8
        @test gFD ≈ gMC rtol = 1e-8
    end
end

# ------------------------------------------------------------------ #
# constant_sed                                                         #
# ------------------------------------------------------------------ #

@testset "constant_sed — type stability" begin
    @testset "JET @test_opt — scalar" begin
        JET.@test_opt CMBForegrounds.constant_sed(150.0)
    end

    @testset "JET @test_opt — vector" begin
        JET.@test_opt CMBForegrounds.constant_sed(NU_TEST)
    end

    @testset "Returns 1.0 for any frequency" begin
        @test CMBForegrounds.constant_sed(90.0)  == 1.0
        @test CMBForegrounds.constant_sed(220.0) == 1.0
        @test all(CMBForegrounds.constant_sed(NU_TEST) .== 1.0)
    end
end
