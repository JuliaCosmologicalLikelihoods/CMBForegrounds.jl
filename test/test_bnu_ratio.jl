"""
Unit tests for Bnu_ratio function

Tests the Planck function ratio B_ν(T)/B_ν0(T) at different frequencies
"""

@testset "Bnu_ratio() Unit Tests" begin

    @testset "Basic Functionality" begin
        # Test with simple values
        ν, ν0, T = 100.0, 50.0, 2.7
        ratio = CMBForegrounds.Bnu_ratio(ν, ν0, T)

        # Basic output tests
        @test ratio isa Number
        @test isfinite(ratio)
        @test ratio > 0  # Planck function ratios should always be positive
    end

    @testset "Mathematical Properties" begin
        # Test with ν = ν0 (should give ratio = 1)
        ratio = CMBForegrounds.Bnu_ratio(143.0, 143.0, 2.7)
        @test ratio ≈ 1.0

        # Test behavior around CMB peak frequency (~160 GHz for T_CMB)
        T = 2.725
        ratio_low = CMBForegrounds.Bnu_ratio(100.0, 143.0, T)
        ratio_ref = CMBForegrounds.Bnu_ratio(143.0, 143.0, T)
        ratio_high = CMBForegrounds.Bnu_ratio(217.0, 143.0, T)

        @test ratio_ref ≈ 1.0
        # At CMB temperatures, the Planck function peaks around 160 GHz
        # So 100 GHz < peak, 143 GHz near peak, 217 GHz > peak
        # The ratio doesn't increase monotonically - it depends on distance from peak
        @test ratio_low < ratio_ref  # 100 GHz is below peak
        @test ratio_high < ratio_ref  # 217 GHz is above peak
    end

    @testset "Consistency with Formula" begin
        # Test that the function matches the mathematical definition
        ν, ν0, T = 100.0, 50.0, 2.7

        # Get dimensionless variables
        r, x, x0 = CMBForegrounds.dimensionless_freq_vars(ν, ν0, T)

        # Manual calculation: r^3 * expm1(x0) / expm1(x)
        expected_ratio = r^3 * expm1(x0) / expm1(x)
        actual_ratio = CMBForegrounds.Bnu_ratio(ν, ν0, T)

        @test actual_ratio ≈ expected_ratio
    end

    @testset "Scaling Properties" begin
        ν, ν0, T = 100.0, 50.0, 2.7

        # If we double the frequency, ratio should change predictably
        ratio1 = CMBForegrounds.Bnu_ratio(ν, ν0, T)
        ratio2 = CMBForegrounds.Bnu_ratio(2 * ν, ν0, T)

        # The ratio scales as r^3 * expm1(x0)/expm1(x)
        # where r doubles and x doubles, so it's not simply 8x
        @test ratio2 > ratio1  # Should increase
        @test ratio2 != 8 * ratio1  # But not simply r^3 scaling due to expm1 terms

        # If we double the reference frequency
        ratio3 = CMBForegrounds.Bnu_ratio(ν, 2 * ν0, T)
        @test ratio3 < ratio1  # Should decrease (lower reference frequency)
    end

    @testset "Type Stability" begin
        # Test with Float64
        ratio = CMBForegrounds.Bnu_ratio(100.0, 50.0, 2.7)
        @test ratio isa Float64

        # Test with Int (should promote to Float64)
        ratio = CMBForegrounds.Bnu_ratio(100, 50, 3)
        @test ratio isa Float64

        # Test with mixed types
        ratio = CMBForegrounds.Bnu_ratio(100, 50.0, 2.7)
        @test ratio isa Float64
    end

    @testset "Edge Cases" begin
        # Test with very small frequencies (Rayleigh-Jeans limit)
        # In R-J limit: B_ν ∝ ν^2 * T, so ratio → (ν/ν0)^2
        ratio_small = CMBForegrounds.Bnu_ratio(0.001, 0.0005, 2.7)
        @test ratio_small ≈ 4.0 rtol = 0.1  # Should approach (ν/ν0)^2 = 4

        # Test with large frequencies (Wien limit)
        # In Wien limit: expm1(x) ≈ exp(x), so ratio → r^3 * exp(x0-x)
        ratio_large = CMBForegrounds.Bnu_ratio(1000.0, 500.0, 0.1)
        @test isfinite(ratio_large)
        @test ratio_large > 0

        # Test with very small temperature (Wien regime)
        ratio_wien = CMBForegrounds.Bnu_ratio(100.0, 50.0, 0.01)
        @test isfinite(ratio_wien)
        @test ratio_wien > 0
    end

    @testset "Physical Consistency" begin
        # Test with realistic CMB values
        T_CMB = CMBForegrounds.T_CMB

        # Planck frequencies
        ratio_100 = CMBForegrounds.Bnu_ratio(100.0, 143.0, T_CMB)
        ratio_143 = CMBForegrounds.Bnu_ratio(143.0, 143.0, T_CMB)
        ratio_217 = CMBForegrounds.Bnu_ratio(217.0, 143.0, T_CMB)
        ratio_353 = CMBForegrounds.Bnu_ratio(353.0, 143.0, T_CMB)

        # Physical expectations - 143 GHz is close to CMB peak (~160 GHz)
        @test ratio_143 ≈ 1.0
        @test ratio_100 < ratio_143  # Below peak
        @test ratio_217 < ratio_143  # Above peak
        @test ratio_353 < ratio_217  # Further above peak, even lower

        # Check reasonable values for CMB physics
        @test 0.3 < ratio_100 < 1.0    # Below peak but not too low
        @test 0.3 < ratio_217 < 1.0    # Above peak, lower than reference
        @test 0.1 < ratio_353 < 0.5    # Far above peak, quite low
    end

    @testset "Numerical Stability" begin
        # Test near x = 0 (should handle expm1 correctly)
        ratio = CMBForegrounds.Bnu_ratio(1e-6, 5e-7, 1000.0)  # Very low freq, high T
        @test isfinite(ratio)
        @test ratio > 0

        # Test for large x values (should not overflow)
        ratio = CMBForegrounds.Bnu_ratio(1000.0, 100.0, 0.1)
        @test isfinite(ratio)
        @test ratio > 0
    end

    @testset "Symmetry Properties" begin
        # Test reciprocal relationship: Bnu_ratio(ν0, ν, T) = 1/Bnu_ratio(ν, ν0, T)
        ν, ν0, T = 100.0, 50.0, 2.7
        ratio_forward = CMBForegrounds.Bnu_ratio(ν, ν0, T)
        ratio_reverse = CMBForegrounds.Bnu_ratio(ν0, ν, T)

        @test ratio_forward * ratio_reverse ≈ 1.0
    end
end
