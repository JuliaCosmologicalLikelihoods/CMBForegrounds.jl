"""
Unit tests for shot_noise_power function

Tests the shot noise power spectrum function that computes:
Dℓ = A_ℓ0 * (ℓ/ℓ0)²

This models the power spectrum of shot noise from discrete point sources.
"""

@testset "shot_noise_power() Unit Tests" begin

    @testset "Basic Functionality" begin
        # Test with typical parameters
        ℓs = [1000, 2000, 3000, 4000, 5000]
        A_ℓ0 = 100.0  # Shot noise amplitude at ℓ0

        D_shot = CMBForegrounds.shot_noise_power(ℓs, A_ℓ0)  # Use default ℓ0=3000

        # Basic output tests
        @test D_shot isa AbstractVector
        @test length(D_shot) == length(ℓs)
        @test all(D_shot .>= 0)  # Shot noise power should be non-negative
        @test all(isfinite.(D_shot))
        @test eltype(D_shot) <: AbstractFloat
    end

    @testset "Mathematical Formula Verification" begin
        # Test the exact formula: Dℓ = A_ℓ0 * (ℓ/ℓ0)²
        ℓs = [1500, 3000, 6000]  # Include the pivot point
        A_ℓ0 = 50.0
        ℓ0 = 3000

        D_shot = CMBForegrounds.shot_noise_power(ℓs, A_ℓ0; ℓ0=ℓ0)

        # Manual calculation
        expected = @. A_ℓ0 * (ℓs / ℓ0)^2
        @test all(D_shot .≈ expected)

        # Test specific values
        @test D_shot[1] ≈ A_ℓ0 * (1500 / 3000)^2  # = A_ℓ0 * 0.25
        @test D_shot[2] ≈ A_ℓ0 * (3000 / 3000)^2  # = A_ℓ0 * 1.0 = A_ℓ0
        @test D_shot[3] ≈ A_ℓ0 * (6000 / 3000)^2  # = A_ℓ0 * 4.0

        @test D_shot[1] ≈ A_ℓ0 / 4
        @test D_shot[2] ≈ A_ℓ0
        @test D_shot[3] ≈ A_ℓ0 * 4
    end

    @testset "Power Law Scaling" begin
        # Test that shot noise follows exact ℓ² scaling
        ℓs = [500, 1000, 2000, 4000]  # Each double the previous
        A_ℓ0 = 100.0
        ℓ0 = 1000  # Use 1000 as pivot for cleaner arithmetic

        D_shot = CMBForegrounds.shot_noise_power(ℓs, A_ℓ0; ℓ0=ℓ0)

        # For ℓ² scaling, doubling ℓ should quadruple the power
        @test D_shot[2] / D_shot[1] ≈ (1000 / 500)^2  # = 4
        @test D_shot[3] / D_shot[2] ≈ (2000 / 1000)^2  # = 4
        @test D_shot[4] / D_shot[3] ≈ (4000 / 2000)^2  # = 4

        # All ratios should be 4 for exact ℓ² scaling
        @test D_shot[2] ≈ 4 * D_shot[1]
        @test D_shot[3] ≈ 4 * D_shot[2]
        @test D_shot[4] ≈ 4 * D_shot[3]
    end

    @testset "Pivot Point Behavior" begin
        # Test that at the pivot point ℓ0, Dℓ equals A_ℓ0
        A_ℓ0 = 75.0
        ℓ0_values = [1000, 3000, 5000]

        for ℓ0 in ℓ0_values
            # Test at the pivot point
            D_at_pivot = CMBForegrounds.shot_noise_power([ℓ0], A_ℓ0; ℓ0=ℓ0)[1]
            @test D_at_pivot ≈ A_ℓ0  # Should equal amplitude at pivot

            # Test at other points relative to this pivot
            ℓs_test = [ℓ0 / 2, ℓ0, 2 * ℓ0]
            D_test = CMBForegrounds.shot_noise_power(ℓs_test, A_ℓ0; ℓ0=ℓ0)

            @test D_test[1] ≈ A_ℓ0 / 4    # At ℓ0/2: (1/2)² = 1/4
            @test D_test[2] ≈ A_ℓ0        # At ℓ0: (1)² = 1
            @test D_test[3] ≈ A_ℓ0 * 4    # At 2*ℓ0: (2)² = 4
        end
    end

    @testset "Amplitude Scaling" begin
        # Test that output scales linearly with amplitude
        ℓs = [1000, 2000, 3000, 4000]
        ℓ0 = 3000

        A1, A2 = 50.0, 150.0
        D1 = CMBForegrounds.shot_noise_power(ℓs, A1; ℓ0=ℓ0)
        D2 = CMBForegrounds.shot_noise_power(ℓs, A2; ℓ0=ℓ0)

        # Should scale exactly linearly with amplitude
        @test all(D2 .≈ (A2 / A1) .* D1)
        @test all(D2 .≈ 3.0 .* D1)  # Since A2/A1 = 150/50 = 3

        # Zero amplitude gives zero result
        D_zero = CMBForegrounds.shot_noise_power(ℓs, 0.0; ℓ0=ℓ0)
        @test all(D_zero .== 0.0)
    end

    @testset "Default Parameters" begin
        # Test default ℓ0=3000 behavior
        ℓs = [1500, 3000, 6000]
        A_ℓ0 = 100.0

        # With default ℓ0
        D_default = CMBForegrounds.shot_noise_power(ℓs, A_ℓ0)

        # With explicit ℓ0=3000
        D_explicit = CMBForegrounds.shot_noise_power(ℓs, A_ℓ0; ℓ0=3000)

        # Should be identical
        @test all(D_default .≈ D_explicit)

        # Check specific values with default ℓ0=3000
        @test D_default[1] ≈ A_ℓ0 * (1500 / 3000)^2  # = A_ℓ0/4
        @test D_default[2] ≈ A_ℓ0 * (3000 / 3000)^2  # = A_ℓ0
        @test D_default[3] ≈ A_ℓ0 * (6000 / 3000)^2  # = A_ℓ0*4
    end

    @testset "Type Stability" begin
        # Test with different input types
        ℓs = [1000, 2000, 3000]
        A_ℓ0 = 100.0

        # Float64 inputs
        D_float = CMBForegrounds.shot_noise_power(ℓs, A_ℓ0)
        @test eltype(D_float) == Float64

        # Integer amplitude - will promote to Float64 due to internal computations
        D_int_A = CMBForegrounds.shot_noise_power(ℓs, 100)  # Integer amplitude
        @test eltype(D_int_A) == Float64  # Promotes to Float64 due to division/multiplication

        # Mixed types
        ℓs_int = [1000, 2000, 3000]  # Integers
        D_mixed = CMBForegrounds.shot_noise_power(ℓs_int, 100.0)  # Float amplitude
        @test eltype(D_mixed) == Float64

        # Integer ℓ0 parameter
        D_int_ℓ0 = CMBForegrounds.shot_noise_power(ℓs, A_ℓ0; ℓ0=3000)  # Integer ℓ0
        @test all(D_int_ℓ0 .≈ D_float)
    end

    @testset "Vector Operations" begin
        # Test with different vector lengths and types

        # Single element
        D_single = CMBForegrounds.shot_noise_power([2000], 100.0; ℓ0=1000)
        @test length(D_single) == 1
        @test D_single[1] ≈ 100.0 * (2000 / 1000)^2  # = 100 * 4 = 400

        # Large vector
        ℓs_large = collect(100:100:10000)  # 100 elements
        D_large = CMBForegrounds.shot_noise_power(ℓs_large, 50.0)
        @test length(D_large) == 100
        @test all(D_large .>= 0)
        @test all(isfinite.(D_large))

        # Check that each element follows the formula
        for i in 1:length(ℓs_large)
            expected = 50.0 * (ℓs_large[i] / 3000)^2
            @test D_large[i] ≈ expected
        end

        # Empty vector
        D_empty = CMBForegrounds.shot_noise_power(Int64[], 100.0)
        @test length(D_empty) == 0
    end

    @testset "Physical Realism - CMB Shot Noise" begin
        # Test with realistic CMB shot noise parameters

        # Typical multipole range for CMB analysis
        ℓs = [500, 1000, 2000, 3000, 5000, 8000]

        # Realistic shot noise amplitudes (in μK²)
        shot_noise_levels = [10.0, 50.0, 200.0, 1000.0]  # Range from low to high noise

        for A_shot in shot_noise_levels
            D_shot = CMBForegrounds.shot_noise_power(ℓs, A_shot)

            # Physical expectations
            @test all(D_shot .>= 0)  # Always non-negative
            @test all(isfinite.(D_shot))
            @test length(D_shot) == length(ℓs)

            # Shot noise should increase with ℓ² - check ordering
            @test issorted(D_shot)  # Should be monotonically increasing

            # At high ℓ, shot noise should be significant
            @test D_shot[end] > D_shot[1]  # Higher ℓ should have higher shot noise

            # The ratio should follow ℓ² scaling
            ratio_expected = (ℓs[end] / ℓs[1])^2
            ratio_actual = D_shot[end] / D_shot[1]
            @test ratio_actual ≈ ratio_expected
        end
    end

    @testset "Mathematical Properties" begin
        # Test mathematical properties of ℓ² scaling

        ℓs = [1000, 3000, 5000]
        A_ℓ0 = 100.0
        ℓ0 = 3000

        D_shot = CMBForegrounds.shot_noise_power(ℓs, A_ℓ0; ℓ0=ℓ0)

        # Homogeneity: scaling all ℓ values should scale result predictably
        scale = 2.0
        ℓs_scaled = scale .* ℓs
        D_scaled = CMBForegrounds.shot_noise_power(ℓs_scaled, A_ℓ0; ℓ0=ℓ0)

        # Since Dℓ ∝ ℓ², scaling ℓ by factor s scales Dℓ by s²
        @test all(D_scaled .≈ scale^2 .* D_shot)

        # Additivity of amplitudes
        A1, A2 = 60.0, 40.0
        D1 = CMBForegrounds.shot_noise_power(ℓs, A1; ℓ0=ℓ0)
        D2 = CMBForegrounds.shot_noise_power(ℓs, A2; ℓ0=ℓ0)
        D_sum = CMBForegrounds.shot_noise_power(ℓs, A1 + A2; ℓ0=ℓ0)

        # Should be additive in amplitude
        @test all(D_sum .≈ D1 .+ D2)
    end

    @testset "Edge Cases" begin
        # Test with extreme but valid parameter values
        ℓs = [100, 1000, 10000]

        # Very large amplitude
        A_large = 1e6
        D_large = CMBForegrounds.shot_noise_power(ℓs, A_large)
        @test all(isfinite.(D_large))
        @test all(D_large .>= 0)

        # Very small amplitude
        A_small = 1e-6
        D_small = CMBForegrounds.shot_noise_power(ℓs, A_small)
        @test all(isfinite.(D_small))
        @test all(D_small .>= 0)
        @test all(D_small .< 1e-3)  # Should be very small

        # Very low multipoles
        ℓs_low = [10, 50, 100]
        D_low = CMBForegrounds.shot_noise_power(ℓs_low, 100.0)
        @test all(isfinite.(D_low))
        @test all(D_low .> 0)
        @test D_low[1] < D_low[2] < D_low[3]  # Should increase

        # Very high multipoles
        ℓs_high = [10000, 50000, 100000]
        D_high = CMBForegrounds.shot_noise_power(ℓs_high, 100.0)
        @test all(isfinite.(D_high))
        @test all(D_high .> 0)

        # Extreme ℓ0 values
        A_ℓ0 = 100.0

        # Very low ℓ0
        D_low_pivot = CMBForegrounds.shot_noise_power([1000], A_ℓ0; ℓ0=10)
        @test isfinite(D_low_pivot[1])
        @test D_low_pivot[1] > A_ℓ0  # Should be much larger since ℓ >> ℓ0

        # Very high ℓ0
        D_high_pivot = CMBForegrounds.shot_noise_power([1000], A_ℓ0; ℓ0=100000)
        @test isfinite(D_high_pivot[1])
        @test D_high_pivot[1] < A_ℓ0  # Should be much smaller since ℓ << ℓ0
    end

    @testset "Numerical Precision" begin
        # Test numerical precision with challenging values

        # High precision constants
        A_precise = π * 100
        ℓ0_precise = 3000
        ℓs_precise = [π * 1000, ℯ * 1000, sqrt(2) * 1000]

        D_precise = CMBForegrounds.shot_noise_power(ℓs_precise, A_precise; ℓ0=ℓ0_precise)

        # Should maintain precision
        @test all(isfinite.(D_precise))
        for i in 1:length(ℓs_precise)
            expected = A_precise * (ℓs_precise[i] / ℓ0_precise)^2
            @test D_precise[i] ≈ expected
        end

        # Very close multipole values
        ℓ_base = 3000.0
        ℓs_close = [ℓ_base - 1e-6, ℓ_base, ℓ_base + 1e-6]
        D_close = CMBForegrounds.shot_noise_power(ℓs_close, 100.0; ℓ0=3000)

        # Middle value should be exactly the amplitude
        @test D_close[2] ≈ 100.0

        # Close values should be very close to the middle value
        @test abs(D_close[1] - 100.0) < 1e-6  # Relaxed tolerance for floating point precision
        @test abs(D_close[3] - 100.0) < 1e-6
    end

    @testset "Implementation Efficiency" begin
        # Test that the efficient implementation works correctly
        ℓs = [1000, 2000, 3000, 4000]
        A_ℓ0 = 100.0
        ℓ0 = 3000

        D_shot = CMBForegrounds.shot_noise_power(ℓs, A_ℓ0; ℓ0=ℓ0)

        # The implementation computes s = A_ℓ0 / (ℓ0 * ℓ0), then ℓs * ℓs * s
        # This should be equivalent to A_ℓ0 * (ℓs / ℓ0)^2
        s_expected = A_ℓ0 / (ℓ0 * ℓ0)
        D_manual = @. ℓs * ℓs * s_expected

        @test all(D_shot .≈ D_manual)

        # Also test against direct formula
        D_direct = @. A_ℓ0 * (ℓs / ℓ0)^2
        @test all(D_shot .≈ D_direct)
    end

    @testset "Comparison with Other Power Spectra" begin
        # Compare shot noise scaling with other typical CMB power spectra

        ℓs = [100, 500, 1000, 3000, 6000]
        A_shot = 100.0

        D_shot = CMBForegrounds.shot_noise_power(ℓs, A_shot)  # ℓ² scaling

        # Shot noise should increase faster than linear with ℓ
        @test D_shot[2] / D_shot[1] > 500 / 100  # More than linear increase
        @test D_shot[3] / D_shot[2] > 1000 / 500  # More than linear increase

        # At high ℓ, shot noise should dominate over typical CMB signal
        # (This is why shot noise is important in high-ℓ CMB analysis)
        @test D_shot[end] > 10 * D_shot[1]  # Much larger at high ℓ

        # The scaling should be exactly quadratic
        for i in 2:length(ℓs)
            expected_ratio = (ℓs[i] / ℓs[1])^2
            actual_ratio = D_shot[i] / D_shot[1]
            @test actual_ratio ≈ expected_ratio
        end
    end

    @testset "Consistency Checks" begin
        # Test internal consistency and relationships

        ℓs = [1500, 3000, 6000]
        A_ℓ0 = 200.0

        # Different ways to compute the same result

        # Method 1: Direct function call
        D1 = CMBForegrounds.shot_noise_power(ℓs, A_ℓ0; ℓ0=3000)

        # Method 2: Using different pivot - should give different results
        D2 = CMBForegrounds.shot_noise_power(ℓs, A_ℓ0; ℓ0=1500)

        # Should give different results (since same amplitude at different pivots)
        @test !(all(D1 .≈ D2))

        # Method 3: Manual computation
        D3 = @. A_ℓ0 * (ℓs / 3000)^2
        @test all(D1 .≈ D3)
    end

    @testset "Realistic CMB Analysis" begin
        # Test with parameters typical of real CMB analysis

        # Multipole range for typical CMB power spectrum analysis
        ℓs = collect(50:50:6000)  # ℓ from 50 to 6000 in steps of 50

        # Shot noise levels for different sensitivity levels
        # (typical values for CMB experiments in μK²)
        sensitivities = [
            ("Planck-like", 100.0),
            ("ACT-like", 50.0),
            ("SO-like", 10.0),
            ("CMB-S4-like", 1.0)
        ]

        for (experiment, noise_level) in sensitivities
            D_shot = CMBForegrounds.shot_noise_power(ℓs, noise_level)

            # Physical expectations for each experiment
            @test all(D_shot .>= 0)
            @test all(isfinite.(D_shot))
            @test issorted(D_shot)  # Should increase with ℓ

            # At ℓ=3000 (the default pivot), should equal the noise level
            idx_3000 = findfirst(x -> x == 3000, ℓs)
            if idx_3000 !== nothing
                @test D_shot[idx_3000] ≈ noise_level
            end

            # At ℓ=6000, should be 4× the noise level
            idx_6000 = findfirst(x -> x == 6000, ℓs)
            if idx_6000 !== nothing
                @test D_shot[idx_6000] ≈ 4 * noise_level
            end
        end
    end
end
