using ADTypes
using ChainRulesCore
using DifferentiationInterface
using FiniteDifferences
using ForwardDiff
using LinearAlgebra
using Mooncake

# Public fixed-beam chromatic bandpass API.
#
# `prepare_fixed_chromatic_bandpass` / `eval_fixed_chromatic_sed_bands` are the
# supported interface for the case where the chromatic beam is a measured
# instrument product rather than an inference parameter. They must be
# numerically identical to the active-beam route and differ only in that the
# beam receives no cotangent in reverse mode.

function _fixed_api_fixture(; n_nu = 9, n_ell = 6, n_bands = 3)
    rng = Xoshiro(0xF1BE)
    ells = collect(1:n_ell)
    nu = collect(range(120.0, 180.0; length = n_nu))
    raws = [RawBand(copy(nu), 0.2 .+ rand(rng, n_nu)) for _ in 1:n_bands]
    beams = [ChromaticBeam(ells, 0.5 .+ rand(rng, n_ell, n_nu)) for _ in 1:n_bands]
    shifts = [0.7, -1.3, 0.4]
    return (; rng, ells, raws, beams, shifts, n_nu, n_ell, n_bands)
end

@testset "Fixed chromatic bandpass — public API is exported" begin
    for name in (:prepare_fixed_chromatic_bandpass, :eval_fixed_chromatic_sed_bands)
        @test isdefined(CMBForegrounds, name)
        @test Base.isexported(CMBForegrounds, name)
    end
    # Documented, so Documenter's `@docs` block cannot silently go stale.
    for name in (:prepare_fixed_chromatic_bandpass, :eval_fixed_chromatic_sed_bands)
        @test !isempty(string(Base.Docs.doc(Base.Docs.Binding(CMBForegrounds, name))))
    end
end

@testset "Fixed chromatic bandpass — primal equals the active-beam route" begin
    fixture = _fixed_api_fixture()
    sed = ModifiedBlackbodySED(150.0, 9.6)
    beta = 1.7

    bands = [shift_and_normalize(raw, shift)
             for (raw, shift) in zip(fixture.raws, fixture.shifts)]

    fixed = [prepare_fixed_chromatic_bandpass(band, beam)
             for (band, beam) in zip(bands, fixture.beams)]
    active = [prepare_chromatic_bandpass(band, beam)
              for (band, beam) in zip(bands, fixture.beams)]

    # The prepared quantities themselves must agree exactly.
    for (a, b) in zip(fixed, active)
        @test a.weights == b.weights
        @test a.denominator == b.denominator
        @test a.band === b.band
        @test a.beam === b.beam
    end

    evaluator = frequency -> sed_weight(sed, frequency, beta)
    fixed_weights = eval_fixed_chromatic_sed_bands(evaluator, fixed)
    active_weights = eval_chromatic_sed_bands(evaluator, active)
    @test size(fixed_weights) == (fixture.n_bands, fixture.n_ell)
    @test fixed_weights == active_weights

    # Against an explicit, independently written quadrature.
    for (index, (band, beam)) in enumerate(zip(bands, fixture.beams))
        for ell in 1:fixture.n_ell
            profile = band.norm_bp .* @view beam.beam[ell, :]
            expected = trapz(band.nu, profile .* evaluator.(band.nu)) /
                       trapz(band.nu, profile)
            @test fixed_weights[index, ell] ≈ expected rtol=1e-12
        end
    end

    # A frequency-independent SED must normalize to one.
    unit_weights = eval_fixed_chromatic_sed_bands(_ -> 1.0, fixed)
    @test all(≈(1.0), unit_weights)

    # A monochromatic band cancels the beam identically.
    mono_band = make_band([150.0], [1.0])
    mono_beam = ChromaticBeam(fixture.ells, rand(fixture.rng, fixture.n_ell, 1))
    mono_fixed = prepare_fixed_chromatic_bandpass(mono_band, mono_beam)
    mono_active = prepare_chromatic_bandpass(mono_band, mono_beam)
    @test mono_fixed.weights === nothing
    @test eval_fixed_chromatic_sed_bands(evaluator, [mono_fixed]) ==
          eval_chromatic_sed_bands(evaluator, [mono_active])

    delta_fixed = prepare_fixed_chromatic_bandpass(DeltaBand(150.0), mono_beam)
    @test delta_fixed.weights === nothing
    @test eval_fixed_chromatic_sed_bands(evaluator, [delta_fixed]) ≈
          fill(evaluator(150.0), 1, fixture.n_ell)
end

@testset "Fixed chromatic bandpass — input validation" begin
    fixture = _fixed_api_fixture()
    band = shift_and_normalize(fixture.raws[1], 0.0)
    mismatched = ChromaticBeam(fixture.ells, ones(fixture.n_ell, fixture.n_nu + 1))
    @test_throws DimensionMismatch prepare_fixed_chromatic_bandpass(band, mismatched)

    zero_beam = ChromaticBeam(fixture.ells, zeros(fixture.n_ell, fixture.n_nu))
    @test_throws DomainError prepare_fixed_chromatic_bandpass(band, zero_beam)

    @test_throws ArgumentError eval_fixed_chromatic_sed_bands(
        identity, PreparedChromaticBandpass[])

    other_grid = ChromaticBeam(collect(1:(fixture.n_ell + 1)),
                               ones(fixture.n_ell + 1, fixture.n_nu))
    inconsistent = [prepare_fixed_chromatic_bandpass(band, fixture.beams[1]),
                    prepare_fixed_chromatic_bandpass(band, other_grid)]
    @test_throws ArgumentError eval_fixed_chromatic_sed_bands(identity, inconsistent)
end

@testset "Fixed chromatic bandpass — the beam receives no cotangent" begin
    fixture = _fixed_api_fixture()
    band = shift_and_normalize(fixture.raws[1], 0.5)
    beam = fixture.beams[1]
    prepared = prepare_fixed_chromatic_bandpass(band, beam)

    # The preparation contracts the beam through `_fixed_beam_product`, and the
    # evaluation through `_fixed_chromatic_ratio`. Both must report the beam as
    # NoTangent for arbitrary cotangents.
    product_cotangent = randn(fixture.rng, fixture.n_ell)
    product, product_pullback = ChainRulesCore.rrule(
        CMBForegrounds._fixed_beam_product, beam.beam, prepared.weights)
    @test product ≈ beam.beam * prepared.weights
    for _ in 1:6
        cotangent = randn(fixture.rng, fixture.n_ell)
        function̄, beam̄, weights̄ = product_pullback(cotangent)
        @test function̄ isa ChainRulesCore.NoTangent
        @test beam̄ isa ChainRulesCore.NoTangent
        @test weights̄ ≈ transpose(beam.beam) * cotangent
        # Exact adjoint identity.
        direction = randn(fixture.rng, length(prepared.weights))
        @test dot(cotangent, beam.beam * direction) ≈ dot(weights̄, direction) rtol=1e-12
    end
    @test product_pullback(ChainRulesCore.Thunk(() -> product_cotangent))[3] ≈
          transpose(beam.beam) * product_cotangent

    sed_values = 0.5 .+ rand(fixture.rng, fixture.n_nu)
    ratio, ratio_pullback = ChainRulesCore.rrule(
        CMBForegrounds._fixed_chromatic_ratio, beam.beam, prepared.weights,
        sed_values, prepared.denominator)
    @test ratio ≈ (beam.beam * (prepared.weights .* sed_values)) ./ prepared.denominator
    for _ in 1:6
        cotangent = randn(fixture.rng, fixture.n_ell)
        function̄, beam̄, weights̄, sed̄, denominator̄ = ratio_pullback(cotangent)
        @test function̄ isa ChainRulesCore.NoTangent
        @test beam̄ isa ChainRulesCore.NoTangent
        scaled = cotangent ./ prepared.denominator
        weighted = transpose(beam.beam) * scaled
        @test weights̄ ≈ sed_values .* weighted
        @test sed̄ ≈ prepared.weights .* weighted
        @test denominator̄ ≈ -scaled .* ratio
    end
end

@testset "Fixed chromatic bandpass — gradients w.r.t. active bandpass shifts" begin
    fixture = _fixed_api_fixture()
    sed = ModifiedBlackbodySED(150.0, 9.6)
    cotangents = randn(Xoshiro(0xF1BF), fixture.n_bands, fixture.n_ell)

    # The beam is fixed, but the bands are shifted and renormalized *inside* the
    # differentiated call, so derivatives with respect to the shifts and the SED
    # parameter must survive.
    function objective(parameters)
        beta = parameters[1]
        shifts = parameters[2:end]
        bands = [shift_and_normalize(raw, shift)
                 for (raw, shift) in zip(fixture.raws, shifts)]
        prepared = [prepare_fixed_chromatic_bandpass(band, beam)
                    for (band, beam) in zip(bands, fixture.beams)]
        weights = eval_fixed_chromatic_sed_bands(
            frequency -> sed_weight(sed, frequency, beta), prepared)
        return dot(cotangents, weights)
    end

    function active_objective(parameters)
        beta = parameters[1]
        shifts = parameters[2:end]
        bands = [shift_and_normalize(raw, shift)
                 for (raw, shift) in zip(fixture.raws, shifts)]
        prepared = [prepare_chromatic_bandpass(band, beam)
                    for (band, beam) in zip(bands, fixture.beams)]
        weights = eval_chromatic_sed_bands(
            frequency -> sed_weight(sed, frequency, beta), prepared)
        return dot(cotangents, weights)
    end

    input = vcat(1.7, fixture.shifts)
    @test objective(input) == active_objective(input)

    forward = DifferentiationInterface.gradient(objective, AutoForwardDiff(), input)

    backend = AutoMooncake(; config=nothing)
    preparation = DifferentiationInterface.prepare_gradient(objective, backend, input)
    reverse = similar(input)
    DifferentiationInterface.gradient!(objective, reverse, preparation, backend, input)

    finite = DifferentiationInterface.gradient(
        objective, AutoFiniteDifferences(; fdm=FiniteDifferences.central_fdm(5, 1)), input)

    @test all(isfinite, forward)
    @test all(!iszero, forward)
    @test reverse ≈ forward rtol=1e-10
    @test reverse ≈ finite rtol=1e-6 atol=1e-8

    # The fixed route must reproduce the active route's gradient for the inputs
    # that are active in both.
    @test forward ≈ DifferentiationInterface.gradient(
        active_objective, AutoForwardDiff(), input) rtol=1e-12

    # The prepared cache is reusable at a moved point and captures nothing.
    moved = input .+ [0.15, -0.4, 0.6, -0.2]
    reused = similar(moved)
    DifferentiationInterface.gradient!(objective, reused, preparation, backend, moved)
    @test reused ≈ DifferentiationInterface.gradient(objective, AutoForwardDiff(), moved) rtol=1e-10
    @test !(reused ≈ reverse)
    repeated = similar(input)
    DifferentiationInterface.gradient!(objective, repeated, preparation, backend, input)
    @test repeated ≈ reverse rtol=1e-12
end
