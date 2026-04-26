"""
Type-stability (JET) + autodiff (DifferentiationInterface) tests for the
bandpass machinery:

  trapz, RawBand, Band, make_band, point_band,
  shift_and_normalize, integrate_sed, integrate_tsz, eval_sed_bands

We exercise:
  - construction of `Band` from a smooth Gaussian passband
  - `point_band` (Dirac-delta) behavior — must equal evaluating the SED
    at the effective frequency
  - `shift_and_normalize` — differentiable wrt the shift parameter
  - `integrate_sed` against a pure SED (`tsz_sed`, `mbb_sed`, `radio_sed`,
    `constant_sed`) over a real band
  - `integrate_tsz` (specialized helper)
  - `eval_sed_bands` over a Vector{Band}

Differentiation backends: ForwardDiff and Mooncake (Float64). Zygote is
not supported on these helpers because of the `Vector{...}(undef, n)`
+ in-place `y[i] = …` pattern needed for type stability — ACT calls
them only via ForwardDiff and Mooncake `@from_chainrules`-wrapped
fused assemblers.
"""

using JET
using ADTypes
import DifferentiationInterface as DI
using ForwardDiff
using Mooncake


# ----------------------------------------------------------------- #
# Helpers                                                              #
# ----------------------------------------------------------------- #

# Synthetic ACT-like passband: Gaussian centered at ν0 with width σ
function _make_gaussian_band(::Type{T}, ν0::Real, σ::Real;
                              n::Int=33, halfwidth::Real=4.0) where T<:Real
    νs = collect(range(T(ν0 - halfwidth*σ), T(ν0 + halfwidth*σ); length=n))
    bp = exp.(-((νs .- T(ν0)) ./ T(σ)).^2 ./ 2)
    return νs, bp
end


# ----------------------------------------------------------------- #
# trapz                                                                #
# ----------------------------------------------------------------- #
@testset "trapz — type stability + AD" begin
    x = collect(0.0:0.1:1.0)
    y = sin.(x)

    @testset "JET @test_opt" begin
        JET.@test_opt CMBForegrounds.trapz(x, y)
    end

    # ∫₀¹ sin(x) dx = 1 − cos(1) ≈ 0.4597
    @test isapprox(CMBForegrounds.trapz(x, y), 1 - cos(1); rtol=1e-2)

    # AD: gradient wrt y is the trapezoidal weight vector
    f_y = yy -> CMBForegrounds.trapz(x, yy)
    g_fd = DI.gradient(f_y, AutoForwardDiff(), y)
    g_mc = DI.gradient(f_y, AutoMooncake(config=nothing), y)
    @test all(isfinite, g_fd)
    @test isapprox(g_fd, g_mc; rtol=1e-10)
end


# ----------------------------------------------------------------- #
# make_band + point_band                                               #
# ----------------------------------------------------------------- #
@testset "make_band / point_band — basics" begin
    νs, bp = _make_gaussian_band(Float64, 150.0, 5.0)
    band = CMBForegrounds.make_band(νs, bp)

    @test band.monofreq == false
    @test length(band.norm_bp) == length(νs)
    # Normalization: ∫ τ̃(ν) dν = 1 by construction
    @test isapprox(CMBForegrounds.trapz(band.nu, band.norm_bp), 1.0; rtol=1e-12)

    pb = CMBForegrounds.point_band(150.0)
    @test pb.monofreq == true
    @test pb.nu_eff == 150.0
    @test length(pb.nu) == 1

    # Integrating constant SED against any band → 1.0 (exact, by construction)
    @test isapprox(CMBForegrounds.integrate_sed(CMBForegrounds.constant_sed, band), 1.0; rtol=1e-12)
    @test CMBForegrounds.integrate_sed(CMBForegrounds.constant_sed, pb) == 1.0

    # JET
    JET.@test_opt CMBForegrounds.make_band(νs, bp)
    JET.@test_opt CMBForegrounds.point_band(150.0)
end


# ----------------------------------------------------------------- #
# integrate_sed — point band must equal scalar SED evaluation         #
# ----------------------------------------------------------------- #
@testset "integrate_sed — point band ≡ scalar evaluation" begin
    pb = CMBForegrounds.point_band(150.0)

    f_tsz = ν -> CMBForegrounds.tsz_sed(ν, 143.0)
    @test CMBForegrounds.integrate_sed(f_tsz, pb) == f_tsz(150.0)

    f_mbb = ν -> CMBForegrounds.mbb_sed(ν, 150.0, 1.6, 19.6)
    @test CMBForegrounds.integrate_sed(f_mbb, pb) == f_mbb(150.0)

    f_rad = ν -> CMBForegrounds.radio_sed(ν, 150.0, -2.5)
    @test CMBForegrounds.integrate_sed(f_rad, pb) == f_rad(150.0)

    @test CMBForegrounds.integrate_sed(CMBForegrounds.constant_sed, pb) == 1.0
end


# ----------------------------------------------------------------- #
# integrate_sed — full band, AD wrt nuisance params                   #
# ----------------------------------------------------------------- #
@testset "integrate_sed — Gaussian band, AD wrt SED params" begin
    νs, bp = _make_gaussian_band(Float64, 220.0, 6.0)
    band = CMBForegrounds.make_band(νs, bp)

    # Differentiate wrt MBB β at fixed temperature
    f_beta = β -> CMBForegrounds.integrate_sed(
        ν -> CMBForegrounds.mbb_sed(ν, 150.0, β[1], 19.6), band)
    x0 = [1.6]

    JET.@test_opt CMBForegrounds.integrate_sed(
        ν -> CMBForegrounds.mbb_sed(ν, 150.0, 1.6, 19.6), band)

    g_fd = DI.gradient(f_beta, AutoForwardDiff(), x0)
    g_mc = DI.gradient(f_beta, AutoMooncake(config=nothing), x0)
    @test isfinite(f_beta(x0))
    @test isfinite(g_fd[1])
    @test isapprox(g_fd, g_mc; rtol=1e-6)

    # Differentiate wrt MBB temperature
    f_temp = T -> CMBForegrounds.integrate_sed(
        ν -> CMBForegrounds.mbb_sed(ν, 150.0, 1.6, T[1]), band)
    x0_T = [19.6]
    g_fd_T = DI.gradient(f_temp, AutoForwardDiff(), x0_T)
    g_mc_T = DI.gradient(f_temp, AutoMooncake(config=nothing), x0_T)
    @test isapprox(g_fd_T, g_mc_T; rtol=1e-6)
end


# ----------------------------------------------------------------- #
# integrate_tsz                                                        #
# ----------------------------------------------------------------- #
@testset "integrate_tsz — equivalent to integrate_sed of tsz_sed" begin
    νs, bp = _make_gaussian_band(Float64, 100.0, 4.0)
    band = CMBForegrounds.make_band(νs, bp)
    pb = CMBForegrounds.point_band(100.0)

    @test CMBForegrounds.integrate_tsz(band, 143.0) ≈
          CMBForegrounds.integrate_sed(ν -> CMBForegrounds.tsz_sed(ν, 143.0), band)

    @test CMBForegrounds.integrate_tsz(pb, 143.0) == CMBForegrounds.tsz_sed(100.0, 143.0)

    JET.@test_opt CMBForegrounds.integrate_tsz(band, 143.0)
end


# ----------------------------------------------------------------- #
# shift_and_normalize — differentiable wrt the shift parameter         #
# ----------------------------------------------------------------- #
@testset "shift_and_normalize — AD wrt shift" begin
    νs, bp = _make_gaussian_band(Float64, 150.0, 5.0)
    raw = CMBForegrounds.RawBand(νs, bp)

    # ν₀ for tSZ; integrate the shifted band's tSZ SED
    f_shift = s -> CMBForegrounds.integrate_sed(
        ν -> CMBForegrounds.tsz_sed(ν, 143.0),
        CMBForegrounds.shift_and_normalize(raw, s[1]))
    x0 = [0.0]

    @test isfinite(f_shift(x0))

    g_fd = DI.gradient(f_shift, AutoForwardDiff(), x0)
    g_mc = DI.gradient(f_shift, AutoMooncake(config=nothing), x0)
    @test isfinite(g_fd[1])
    @test isapprox(g_fd, g_mc; rtol=1e-6)

    # Zero-shift identity: result equals make_band(ν, bp) integrated
    band0 = CMBForegrounds.make_band(νs, bp)
    @test isapprox(f_shift(x0), CMBForegrounds.integrate_sed(
        ν -> CMBForegrounds.tsz_sed(ν, 143.0), band0); rtol=1e-12)
end


# ----------------------------------------------------------------- #
# eval_sed_bands                                                       #
# ----------------------------------------------------------------- #
@testset "eval_sed_bands — vector of bands" begin
    νs1, bp1 = _make_gaussian_band(Float64, 100.0, 4.0)
    νs2, bp2 = _make_gaussian_band(Float64, 150.0, 5.0)
    νs3, bp3 = _make_gaussian_band(Float64, 220.0, 6.0)
    bands = [CMBForegrounds.make_band(νs1, bp1),
             CMBForegrounds.make_band(νs2, bp2),
             CMBForegrounds.make_band(νs3, bp3)]

    # Mixing point bands and full bands
    bands_mixed = [CMBForegrounds.point_band(100.0),
                   CMBForegrounds.make_band(νs2, bp2),
                   CMBForegrounds.point_band(220.0)]

    f_tsz = ν -> CMBForegrounds.tsz_sed(ν, 143.0)
    out      = CMBForegrounds.eval_sed_bands(f_tsz, bands)
    out_mix  = CMBForegrounds.eval_sed_bands(f_tsz, bands_mixed)
    @test length(out) == 3
    @test all(isfinite, out)
    @test out_mix[1] == CMBForegrounds.tsz_sed(100.0, 143.0)
    @test out_mix[3] == CMBForegrounds.tsz_sed(220.0, 143.0)
    @test out_mix[2] ≈ out[2]

    JET.@test_opt CMBForegrounds.eval_sed_bands(f_tsz, bands)

    # AD wrt ν₀: differentiate sum of integrated SEDs
    f_nu0 = nu0 -> sum(CMBForegrounds.eval_sed_bands(
        ν -> CMBForegrounds.tsz_sed(ν, nu0[1]), bands))
    x0 = [143.0]
    g_fd = DI.gradient(f_nu0, AutoForwardDiff(), x0)
    g_mc = DI.gradient(f_nu0, AutoMooncake(config=nothing), x0)
    @test isfinite(g_fd[1])
    @test isapprox(g_fd, g_mc; rtol=1e-6)
end
