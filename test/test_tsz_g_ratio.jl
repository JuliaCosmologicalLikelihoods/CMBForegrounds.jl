"""
Unit tests for tsz_g_ratio function

Tests the tSZ (thermal Sunyaev-Zel'dovich) spectral function ratio g(ν)/g(ν0)
where g(x) = x(1 + 2/expm1(x)) - 4
"""

@testset "tsz_g_ratio() Unit Tests" begin

    @testset "Basic Functionality" begin
        # Test with simple values
        ν, ν0, T = 100.0, 50.0, 2.7
        ratio = CMBForegrounds.tsz_g_ratio(ν, ν0, T)

        # Basic output tests
        @test ratio isa Number
        @test isfinite(ratio)
        # Note: tSZ g_ratio can be negative, so no positivity requirement
    end

    @testset "Mathematical Properties" begin
        # Test with ν = ν0 (should give ratio = 1)
        ratio = CMBForegrounds.tsz_g_ratio(143.0, 143.0, 2.7)
        @test ratio ≈ 1.0

        # Test frequency ordering at CMB temperatures
        T = CMBForegrounds.T_CMB
        ratio_100 = CMBForegrounds.tsz_g_ratio(100.0, 143.0, T)
        ratio_143 = CMBForegrounds.tsz_g_ratio(143.0, 143.0, T)
        ratio_217 = CMBForegrounds.tsz_g_ratio(217.0, 143.0, T)
        ratio_353 = CMBForegrounds.tsz_g_ratio(353.0, 143.0, T)

        @test ratio_143 ≈ 1.0
        @test ratio_100 > ratio_143    # Lower freq has higher positive value
        @test abs(ratio_217) < 0.1     # 217 GHz is near the tSZ null
        @test ratio_353 < 0            # Higher freq becomes negative
    end

    @testset "Consistency with Formula" begin
        # Test that the function matches the mathematical definition
        ν, ν0, T = 100.0, 50.0, 2.7

        # Get dimensionless variables
        r, x, x0 = CMBForegrounds.dimensionless_freq_vars(ν, ν0, T)

        # Manual calculation of g functions
        g0 = x0 * (1 + 2 / expm1(x0)) - 4
        g = x * (1 + 2 / expm1(x)) - 4
        expected_ratio = g / g0
        actual_ratio = CMBForegrounds.tsz_g_ratio(ν, ν0, T)

        @test actual_ratio ≈ expected_ratio
    end

    @testset "tSZ Null Frequency" begin
        # Test behavior around the tSZ null frequency (~217 GHz for CMB)
        T_CMB = CMBForegrounds.T_CMB

        # Test frequencies around the null
        ratio_216 = CMBForegrounds.tsz_g_ratio(216.0, 143.0, T_CMB)
        ratio_217 = CMBForegrounds.tsz_g_ratio(217.0, 143.0, T_CMB)
        ratio_218 = CMBForegrounds.tsz_g_ratio(218.0, 143.0, T_CMB)

        # Near the null, the ratio should be very small
        @test abs(ratio_217) < 0.1

        # Should cross zero around 217-218 GHz
        @test ratio_216 < 0.1    # Just below null
        @test ratio_218 > -0.1   # Just above null (can be slightly positive)

        # The sign should change around the null
        @test sign(ratio_216) != sign(ratio_218) || abs(ratio_216) < 0.01 || abs(ratio_218) < 0.01
    end

    @testset "Scaling Properties" begin
        ν, ν0, T = 100.0, 50.0, 2.7

        # If we double the frequency, ratio should change
        ratio1 = CMBForegrounds.tsz_g_ratio(ν, ν0, T)
        ratio2 = CMBForegrounds.tsz_g_ratio(2 * ν, ν0, T)

        @test ratio2 != ratio1  # Should be different

        # If we double the reference frequency
        ratio3 = CMBForegrounds.tsz_g_ratio(ν, 2 * ν0, T)
        @test ratio3 != ratio1  # Should change
    end

    @testset "Type Stability" begin
        # Test with Float64
        ratio = CMBForegrounds.tsz_g_ratio(100.0, 50.0, 2.7)
        @test ratio isa Float64

        # Test with Int (should promote to Float64)
        ratio = CMBForegrounds.tsz_g_ratio(100, 50, 3)
        @test ratio isa Float64

        # Test with mixed types
        ratio = CMBForegrounds.tsz_g_ratio(100, 50.0, 2.7)
        @test ratio isa Float64
    end

    @testset "Edge Cases" begin
        # Test with very small frequencies (Rayleigh-Jeans limit)
        # In R-J limit: g(x) ≈ x(1 + 2/x) - 4 = x + 2 - 4 = x - 2
        ratio_small = CMBForegrounds.tsz_g_ratio(0.001, 0.0005, 10.0)
        @test isfinite(ratio_small)

        # Test with large frequencies (Wien limit)
        ratio_large = CMBForegrounds.tsz_g_ratio(1000.0, 500.0, 0.1)
        @test isfinite(ratio_large)
        @test ratio_large > 0  # In Wien limit, both g and g0 are large positive

        # Test with very small temperature (Wien regime)
        ratio_wien = CMBForegrounds.tsz_g_ratio(100.0, 50.0, 0.01)
        @test isfinite(ratio_wien)
    end

    @testset "Physical Consistency" begin
        # Test with realistic CMB values
        T_CMB = CMBForegrounds.T_CMB

        # Planck frequencies
        ratio_30 = CMBForegrounds.tsz_g_ratio(30.0, 143.0, T_CMB)
        ratio_100 = CMBForegrounds.tsz_g_ratio(100.0, 143.0, T_CMB)
        ratio_143 = CMBForegrounds.tsz_g_ratio(143.0, 143.0, T_CMB)
        ratio_217 = CMBForegrounds.tsz_g_ratio(217.0, 143.0, T_CMB)
        ratio_353 = CMBForegrounds.tsz_g_ratio(353.0, 143.0, T_CMB)

        # Physical expectations for tSZ effect
        @test ratio_143 ≈ 1.0
        @test ratio_30 > ratio_100 > ratio_143  # Decreasing toward reference
        @test abs(ratio_217) < 0.1              # Near null
        @test ratio_353 < -1.0                  # Strongly negative at high freq

        # Check reasonable ranges for CMB physics
        @test 1.0 < ratio_30 < 3.0     # Positive but reasonable
        @test 1.0 < ratio_100 < 2.0    # Between reference and low freq
        @test -15.0 < ratio_353 < -1.0  # Negative but not too extreme
    end

    @testset "Numerical Stability" begin
        # Test near x = 0 (should handle expm1 correctly)
        ratio = CMBForegrounds.tsz_g_ratio(1e-6, 5e-7, 1000.0)  # Very low freq, high T
        @test isfinite(ratio)

        # Test for large x values (should not overflow)
        ratio = CMBForegrounds.tsz_g_ratio(1000.0, 100.0, 0.1)
        @test isfinite(ratio)

        # Test when g0 might be small (but not zero)
        # Find a frequency where g0 is small but non-zero
        T_CMB = CMBForegrounds.T_CMB
        ratio = CMBForegrounds.tsz_g_ratio(100.0, 217.0, T_CMB)  # Reference near null
        @test isfinite(ratio)
    end

    @testset "Symmetry Properties" begin
        # Test reciprocal relationship behavior
        ν, ν0, T = 100.0, 50.0, 2.7
        ratio_forward = CMBForegrounds.tsz_g_ratio(ν, ν0, T)
        ratio_reverse = CMBForegrounds.tsz_g_ratio(ν0, ν, T)

        # For tSZ g_ratio: g(ν)/g(ν0) and g(ν0)/g(ν) are reciprocals
        @test ratio_forward * ratio_reverse ≈ 1.0
    end

    @testset "Temperature Dependence" begin
        # Test how the ratio changes with temperature
        ν, ν0 = 100.0, 143.0

        ratio_lowT = CMBForegrounds.tsz_g_ratio(ν, ν0, 1.0)
        ratio_midT = CMBForegrounds.tsz_g_ratio(ν, ν0, 2.7)
        ratio_highT = CMBForegrounds.tsz_g_ratio(ν, ν0, 10.0)

        # All should be finite
        @test all(isfinite.([ratio_lowT, ratio_midT, ratio_highT]))

        # Temperature affects the spectral function through x = hν/kT
        # Lower T means higher x values, changing the spectral shape
        @test ratio_lowT != ratio_midT
        @test ratio_midT != ratio_highT
    end

    @testset "Limiting Behavior" begin
        # Test Rayleigh-Jeans limit (low frequency, high temperature)
        # g(x) ≈ x + 2 - 4 = x - 2 for small x
        ν, ν0, T = 0.1, 0.05, 100.0  # Very low freq, high temp
        ratio_rj = CMBForegrounds.tsz_g_ratio(ν, ν0, T)

        # In R-J limit, g(x) ≈ x - 2 ≈ -2, so ratio ≈ (x-2)/(x0-2) ≈ 1
        @test ratio_rj ≈ 1.0 rtol = 0.1

        # Test Wien limit (high frequency, low temperature)
        # g(x) ≈ x - 4 for large x
        ν, ν0, T = 1000.0, 500.0, 0.01  # Very high freq, low temp
        ratio_wien = CMBForegrounds.tsz_g_ratio(ν, ν0, T)

        # In Wien limit, g(x) ≈ x - 4, ratio ≈ (x-4)/(x0-4) ≈ x/x0 = ν/ν0 = 2
        @test isfinite(ratio_wien)
        @test abs(ratio_wien - 2.0) < 0.1  # Should be close to 2 in Wien limit
    end

    @testset "Sign Changes" begin
        # Test that the function correctly handles sign changes
        T_CMB = CMBForegrounds.T_CMB

        # Low frequency (positive g)
        ratio_low = CMBForegrounds.tsz_g_ratio(30.0, 143.0, T_CMB)
        @test ratio_low > 0

        # High frequency (negative g)
        ratio_high = CMBForegrounds.tsz_g_ratio(353.0, 143.0, T_CMB)
        @test ratio_high < 0

        # Both g and g0 negative (ratio should be positive)
        ratio_both_neg = CMBForegrounds.tsz_g_ratio(353.0, 300.0, T_CMB)
        @test ratio_both_neg > 0  # Both g and g0 are negative, so ratio is positive

        # g positive, g0 negative (ratio should be negative)
        ratio_mixed = CMBForegrounds.tsz_g_ratio(30.0, 353.0, T_CMB)
        @test ratio_mixed < 0  # g positive, g0 negative
    end
end
