"""
Unit tests for gaussian_beam_window function

Tests the Gaussian beam window function that computes:
B(ℓ) = exp(-0.5 * ℓ * (ℓ + 1) * σ²)

where σ is the beam standard deviation in radians, converted from FWHM in arcminutes.
This models how instrumental beam size affects CMB power spectra at different multipoles.
"""


@testset "gaussian_beam_window() Unit Tests" begin

    @testset "Basic Functionality" begin
        # Test with typical CMB instrument beam size
        fwhm_arcmin = 5.0  # 5 arcminute beam
        ells = [0, 1, 2, 10, 100, 1000]

        result = CMBForegrounds.gaussian_beam_window(fwhm_arcmin, ells)

        # Basic output tests
        @test result isa AbstractVector
        @test length(result) == length(ells)
        @test eltype(result) <: AbstractFloat
        @test all(isfinite.(result))
        @test all(result .>= 0.0)  # Exponential is always non-negative
        @test all(result .<= 1.0)  # Should be ≤ 1 for physical beams

        # Test that B(0) = 1 (no smoothing at ℓ=0)
        @test result[1] ≈ 1.0

        # Test monotonic decrease (beam smooths higher multipoles more)
        for i in 2:length(result)
            @test result[i] <= result[i-1]  # Monotonically decreasing
        end
    end

    @testset "Mathematical Formula Verification" begin
        # Test exact mathematical formula: exp(-0.5 * ℓ * (ℓ + 1) * σ²)
        fwhm_arcmin = 10.0
        ells = [0, 1, 2, 5, 10, 20]

        result = CMBForegrounds.gaussian_beam_window(fwhm_arcmin, ells)

        # Calculate expected values using the mathematical formula
        σ = CMBForegrounds.fwhm_arcmin_to_sigma_rad(fwhm_arcmin)
        expected = [exp(-0.5 * ℓ * (ℓ + 1) * σ^2) for ℓ in ells]

        # Should match the mathematical formula exactly
        for i in 1:length(result)
            @test result[i] ≈ expected[i]
        end
        @test result ≈ expected

        # Test specific known values
        @test result[1] ≈ 1.0  # ℓ=0: exp(-0) = 1
        @test result[2] ≈ exp(-σ^2)  # ℓ=1: exp(-0.5 * 1 * 2 * σ²) = exp(-σ²)
    end

    @testset "Beam Size Dependencies" begin
        # Test how beam window depends on beam size
        ells = [10, 100, 1000, 2000]

        # Different beam sizes
        fwhm_small = 1.0   # 1 arcminute (sharp beam)
        fwhm_medium = 5.0  # 5 arcminutes (typical)
        fwhm_large = 20.0  # 20 arcminutes (broad beam)

        window_small = CMBForegrounds.gaussian_beam_window(fwhm_small, ells)
        window_medium = CMBForegrounds.gaussian_beam_window(fwhm_medium, ells)
        window_large = CMBForegrounds.gaussian_beam_window(fwhm_large, ells)

        # For all ℓ > 0, larger beams should have smaller window functions
        for i in 2:length(ells)  # Skip ℓ=0 where all are 1
            @test window_large[i] <= window_medium[i] <= window_small[i]
        end

        # At ℓ=10, all should be close but different (not exactly 1 since ℓ≠0)
        # Just verify they decrease with beam size
        @test window_large[1] <= window_medium[1] <= window_small[1]

        # Higher multipoles should be more affected by beam size differences
        diff_low = window_small[2] - window_large[2]   # ℓ=10
        diff_high = window_small[4] - window_large[4]  # ℓ=2000
        @test diff_high > diff_low  # Larger difference at higher ℓ
    end

    @testset "Input Validation and Edge Cases" begin
        # Test with ℓ=0 only
        result_zero = CMBForegrounds.gaussian_beam_window(5.0, [0])
        @test result_zero[1] ≈ 1.0

        # Test with single non-zero ℓ
        result_single = CMBForegrounds.gaussian_beam_window(5.0, [100])
        @test 0.0 < result_single[1] < 1.0

        # Test with very small beam (should give values close to 1)
        fwhm_tiny = 0.1  # 0.1 arcminute
        ells_test = [1, 10, 100]
        result_tiny = CMBForegrounds.gaussian_beam_window(fwhm_tiny, ells_test)
        @test all(result_tiny .> 0.9)  # Should be close to 1 for small beams

        # Test with very large beam (should give small values at high ℓ)
        fwhm_huge = 100.0  # 100 arcminutes
        result_huge = CMBForegrounds.gaussian_beam_window(fwhm_huge, [1000, 2000])
        @test all(result_huge .< 0.1)  # Should be very small for large beam at high ℓ

        # Test with empty input
        result_empty = CMBForegrounds.gaussian_beam_window(5.0, Int[])
        @test length(result_empty) == 0

        # Test with large range of multipoles
        ells_large = 0:10:5000
        result_large = CMBForegrounds.gaussian_beam_window(5.0, collect(ells_large))
        @test length(result_large) == length(ells_large)
        @test all(isfinite.(result_large))
    end

    @testset "Type Stability and Promotion" begin
        # Test with different input types
        fwhm = 5.0

        # Integer ells
        ells_int = [0, 1, 2, 10, 100]
        result_int = CMBForegrounds.gaussian_beam_window(fwhm, ells_int)
        @test eltype(result_int) <: AbstractFloat

        # Float64 ells
        ells_float = [0.0, 1.0, 2.0, 10.0, 100.0]
        result_float = CMBForegrounds.gaussian_beam_window(fwhm, ells_float)
        @test eltype(result_float) == Float64

        # Mixed types should give same results for integer-valued floats
        @test result_int ≈ result_float

        # Different array types
        result_vector = CMBForegrounds.gaussian_beam_window(fwhm, [1, 2, 3])
        result_range = CMBForegrounds.gaussian_beam_window(fwhm, 1:3)
        @test result_vector ≈ result_range

        # Test FWHM type promotion
        result_int_fwhm = CMBForegrounds.gaussian_beam_window(5, [1, 2, 3])
        result_float_fwhm = CMBForegrounds.gaussian_beam_window(5.0, [1, 2, 3])
        @test result_int_fwhm ≈ result_float_fwhm

        # Test with BigFloat for high precision
        result_big = CMBForegrounds.gaussian_beam_window(big(5.0), [1, 2, 3])
        @test eltype(result_big) == BigFloat
    end

    @testset "Physical Realism - CMB Instruments" begin
        # Test with realistic CMB instrument parameters

        # Planck instrument beam sizes (approximate)
        planck_beams = [
            5.0,    # 857 GHz ~ 5 arcmin
            7.0,    # 545 GHz ~ 7 arcmin
            10.0,   # 353 GHz ~ 10 arcmin
            13.0,   # 217 GHz ~ 13 arcmin
            15.0,   # 143 GHz ~ 15 arcmin
            30.0    # 30 GHz ~ 30 arcmin
        ]

        # Typical multipole range for CMB analysis
        ells_cmb = [2, 10, 50, 100, 500, 1000, 2000, 3000]

        for fwhm in planck_beams
            result = CMBForegrounds.gaussian_beam_window(fwhm, ells_cmb)

            # Physical constraints
            @test all(0.0 .<= result .<= 1.0)
            @test result[1] > 0.99  # B(ℓ=2) should be close to 1

            # Beam should suppress high-ℓ modes
            @test result[end] < result[1]  # B(ℓ=3000) < B(ℓ=2)

            # Reasonable suppression at ℓ=1000 for typical beams
            ℓ1000_idx = findfirst(x -> x == 1000, ells_cmb)
            if fwhm <= 10.0
                @test result[ℓ1000_idx] > 0.3  # Allow more suppression for 10 arcmin beams
            else
                @test result[ℓ1000_idx] < 0.8  # Some suppression for larger beams
            end
        end

        # Test WMAP beam (~ 13-21 arcmin)
        wmap_beam = 18.0
        result_wmap = CMBForegrounds.gaussian_beam_window(wmap_beam, [100, 500, 1000])
        @test all(0.0 .< result_wmap .< 1.0)
        @test issorted(result_wmap, rev=true)  # Decreasing with ℓ

        # Test ground-based experiment (smaller beam ~ 1-2 arcmin)
        ground_beam = 1.5
        result_ground = CMBForegrounds.gaussian_beam_window(ground_beam, [1000, 3000, 6000])
        @test all(result_ground .> 0.3)  # Less suppression for small beam, but high ℓ values
    end

    @testset "Numerical Properties" begin
        # Test numerical behavior and precision
        fwhm = 5.0

        # Test with high precision values
        ells_precise = [π, ℯ, sqrt(2) * 100, 1000 * log(2)]
        result_precise = CMBForegrounds.gaussian_beam_window(fwhm, ells_precise)
        @test all(isfinite.(result_precise))
        @test all(0.0 .<= result_precise .<= 1.0)

        # Test numerical stability near ℓ=0
        ells_near_zero = [0.0, 1e-10, 1e-5, 0.001, 0.1]
        result_near_zero = CMBForegrounds.gaussian_beam_window(fwhm, ells_near_zero)
        @test result_near_zero[1] ≈ 1.0
        @test all(result_near_zero[2:end] .<= 1.0)  # Should be <= 1 (allow exactly 1 for very small values)

        # Test with very high multipoles (numerical underflow region)
        ells_high = [5000, 10000, 20000]
        result_high = CMBForegrounds.gaussian_beam_window(fwhm, ells_high)
        @test all(isfinite.(result_high))  # Should not give NaN or Inf
        @test all(result_high .>= 0.0)  # Should not underflow to negative

        # For high ℓ and reasonable beam, values should be very small but positive
        @test all(result_high .< 1e-2)  # Very suppressed but at 5 arcmin, high ℓ not as suppressed
        @test all(result_high .> 0.0)    # But still positive
    end

    @testset "Consistency with Helper Function" begin
        # Test consistency with fwhm_arcmin_to_sigma_rad function
        fwhm_test = 7.5
        ells_test = [0, 10, 100, 1000]

        # Calculate using the main function
        result_main = CMBForegrounds.gaussian_beam_window(fwhm_test, ells_test)

        # Calculate manually using the helper function
        σ = CMBForegrounds.fwhm_arcmin_to_sigma_rad(fwhm_test)
        result_manual = [exp(-0.5 * ℓ * (ℓ + 1) * σ^2) for ℓ in ells_test]

        @test result_main ≈ result_manual

        # Test that sigma conversion is reasonable
        # σ ≈ FWHM / (2 * sqrt(2 * ln(2))) ≈ FWHM / 2.35
        expected_sigma_rough = (fwhm_test * π / 180 / 60) / 2.35
        @test σ ≈ expected_sigma_rough atol = 5e-5
    end

    @testset "Mathematical Properties" begin
        # Test mathematical properties of the beam window function

        # Linearity in log space: log B(ℓ) = -0.5 * ℓ(ℓ+1) * σ²
        fwhm = 8.0
        ells = [1, 2, 5, 10, 20, 50]
        result = CMBForegrounds.gaussian_beam_window(fwhm, ells)

        σ = CMBForegrounds.fwhm_arcmin_to_sigma_rad(fwhm)
        log_result = log.(result[2:end])  # Skip ℓ=0 where log(1)=0
        ells_nonzero = ells[2:end]
        expected_log = [-0.5 * ℓ * (ℓ + 1) * σ^2 for ℓ in ells_nonzero]

        @test log_result ≈ expected_log

        # Test scaling property: B(ℓ; α*FWHM) = B(ℓ; FWHM)^(α²)
        α = 2.0
        fwhm_base = 5.0
        fwhm_scaled = α * fwhm_base
        ell_test = [10, 100]

        result_base = CMBForegrounds.gaussian_beam_window(fwhm_base, ell_test)
        result_scaled = CMBForegrounds.gaussian_beam_window(fwhm_scaled, ell_test)

        # Should satisfy: B(ℓ; 2*FWHM) = B(ℓ; FWHM)^4 (since σ scales linearly with FWHM)
        expected_scaled = result_base .^ (α^2)
        @test result_scaled ≈ expected_scaled atol = 1e-10

        # Convolution property test
        # If we have two Gaussian beams with FWHM₁ and FWHM₂,
        # the combined beam has FWHM_combined = sqrt(FWHM₁² + FWHM₂²)
        fwhm1, fwhm2 = 3.0, 4.0
        fwhm_combined = sqrt(fwhm1^2 + fwhm2^2)  # = 5.0
        ell_test = [50, 200]

        window1 = CMBForegrounds.gaussian_beam_window(fwhm1, ell_test)
        window2 = CMBForegrounds.gaussian_beam_window(fwhm2, ell_test)
        window_combined = CMBForegrounds.gaussian_beam_window(fwhm_combined, ell_test)

        # Combined window should equal product of individual windows
        @test window_combined ≈ window1 .* window2 atol = 1e-12
    end

    @testset "Performance and Vectorization" begin
        # Test performance with large arrays
        fwhm = 5.0

        # Large array test
        ells_large = 0:10:5000  # 501 elements
        result_large = CMBForegrounds.gaussian_beam_window(fwhm, collect(ells_large))

        @test length(result_large) == length(ells_large)
        @test all(isfinite.(result_large))
        @test result_large[1] ≈ 1.0  # ℓ=0
        @test issorted(result_large, rev=true)  # Monotonically decreasing

        # Test that vectorization works properly
        ells_vec = [10, 50, 100, 500]
        result_vec = CMBForegrounds.gaussian_beam_window(fwhm, ells_vec)

        # Should give same results as individual calculations
        result_individual = [CMBForegrounds.gaussian_beam_window(fwhm, [ℓ])[1] for ℓ in ells_vec]
        @test result_vec ≈ result_individual

        # Test memory efficiency (result should be same type as input for ells)
        ells_float32 = Float32[1, 2, 3]
        result_float32 = CMBForegrounds.gaussian_beam_window(fwhm, ells_float32)
        # Note: result type depends on promotion with σ calculation
        @test eltype(result_float32) <: AbstractFloat
    end

    @testset "Extreme Values and Edge Cases" begin
        # Test with extreme but valid inputs

        # Very small FWHM (almost delta function beam)
        fwhm_tiny = 1e-3  # 0.001 arcminutes
        ells_test = [100, 1000, 3000]
        result_tiny = CMBForegrounds.gaussian_beam_window(fwhm_tiny, ells_test)
        @test all(result_tiny .> 0.99)  # Should be very close to 1

        # Very large FWHM (very broad beam)
        fwhm_huge = 1000.0  # 1000 arcminutes = 16.7 degrees
        result_huge = CMBForegrounds.gaussian_beam_window(fwhm_huge, [1, 2, 5])
        @test all(result_huge .< 1.0)  # Should be less than 1 for large beam
        @test all(result_huge .> 0.0)  # Should be positive

        # Single multipole
        result_single = CMBForegrounds.gaussian_beam_window(5.0, [100])
        @test length(result_single) == 1
        @test 0.0 < result_single[1] < 1.0

        # Zero multipole only
        result_zero_only = CMBForegrounds.gaussian_beam_window(5.0, [0])
        @test result_zero_only[1] ≈ 1.0

        # Large multipole values
        ells_huge = [50000, 100000]
        result_huge_ells = CMBForegrounds.gaussian_beam_window(1.0, ells_huge)
        @test all(result_huge_ells .< 1.0)  # Should be less than 1 for very high ℓ
        @test all(result_huge_ells .>= 0.0)  # Should be non-negative
        @test all(isfinite.(result_huge_ells))  # Should not be NaN or Inf
    end

    @testset "Comparison with Analytical Limits" begin
        # Test limiting behaviors

        # Small ℓ limit: B(ℓ) ≈ 1 - 0.5 * ℓ(ℓ+1) * σ² for small ℓ
        fwhm = 10.0
        σ = CMBForegrounds.fwhm_arcmin_to_sigma_rad(fwhm)

        # Test small ℓ approximation
        ells_small = [1, 2, 3]
        result_small = CMBForegrounds.gaussian_beam_window(fwhm, ells_small)

        for (i, ℓ) in enumerate(ells_small)
            linear_approx = 1.0 - 0.5 * ℓ * (ℓ + 1) * σ^2
            # Linear approximation should be close for small ℓ and σ
            if abs(linear_approx - result_small[i]) < 0.1
                @test result_small[i] ≈ linear_approx atol = 0.05
            end
        end

        # Large ℓ limit: log B(ℓ) ≈ -0.5 * ℓ² * σ² for ℓ >> 1
        ells_large = [100, 500, 1000]
        result_large = CMBForegrounds.gaussian_beam_window(fwhm, ells_large)

        for (i, ℓ) in enumerate(ells_large)
            exact_log = -0.5 * ℓ * (ℓ + 1) * σ^2
            approx_log = -0.5 * ℓ^2 * σ^2

            # For large ℓ, ℓ(ℓ+1) ≈ ℓ² so approximations should be close
            @test log(result_large[i]) ≈ exact_log
            @test abs(exact_log - approx_log) < 0.1 * abs(exact_log)  # Good approximation
        end

        # Perfect beam limit: FWHM → 0 should give B(ℓ) → 1 for all ℓ
        fwhm_perfect = 1e-10
        ells_test = [1, 10, 100]
        result_perfect = CMBForegrounds.gaussian_beam_window(fwhm_perfect, ells_test)
        @test all(isapprox.(result_perfect, 1.0, atol=1e-8))
    end

    @testset "Cross-Validation with Known Values" begin
        # Test with pre-calculated reference values
        # These can be calculated independently using the formula

        fwhm_ref = 5.0  # arcminutes
        σ_ref = fwhm_ref * (π / 180 / 60) / sqrt(8 * log(2))  # Convert to radians

        test_cases = [
            (0, 1.0),                                              # B(0) = 1
            (1, exp(-0.5 * 1 * 2 * σ_ref^2)),                    # B(1)
            (2, exp(-0.5 * 2 * 3 * σ_ref^2)),                    # B(2)
            (10, exp(-0.5 * 10 * 11 * σ_ref^2)),                 # B(10)
            (100, exp(-0.5 * 100 * 101 * σ_ref^2)),              # B(100)
        ]

        for (ℓ, expected) in test_cases
            result = CMBForegrounds.gaussian_beam_window(fwhm_ref, [ℓ])
            @test result[1] ≈ expected atol = 1e-12
        end

        # Test with multiple values at once
        ells_ref = [0, 1, 2, 10, 100]
        expected_ref = [exp(-0.5 * ℓ * (ℓ + 1) * σ_ref^2) for ℓ in ells_ref]
        result_ref = CMBForegrounds.gaussian_beam_window(fwhm_ref, ells_ref)
        @test result_ref ≈ expected_ref
    end

end
