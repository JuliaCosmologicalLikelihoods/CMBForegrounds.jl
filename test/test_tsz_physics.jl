"""
Unit tests for CMBForegrounds.jl fundamental functions

Starting with the most basic building block: dimensionless_freq_vars()
"""

@testset "dimensionless_freq_vars() Unit Tests" begin

    @testset "Basic Functionality" begin
        # Test with simple values
        ν, ν0, T = 100.0, 50.0, 2.7
        r, x, x0 = CMBForegrounds.dimensionless_freq_vars(ν, ν0, T)

        # Basic output tests
        @test r isa Number
        @test x isa Number
        @test x0 isa Number
        @test all(isfinite.([r, x, x0]))  # All outputs should be finite
    end

    @testset "Mathematical Properties" begin
        # Test frequency ratio r = ν/ν0
        @test CMBForegrounds.dimensionless_freq_vars(100.0, 50.0, 2.7)[1] ≈ 2.0  # r should be exactly 2
        @test CMBForegrounds.dimensionless_freq_vars(143.0, 143.0, 2.7)[1] ≈ 1.0  # r should be exactly 1
        @test CMBForegrounds.dimensionless_freq_vars(50.0, 100.0, 2.7)[1] ≈ 0.5  # r should be exactly 0.5

        # Test that x = r * x0
        ν, ν0, T = 100.0, 50.0, 2.7
        r, x, x0 = CMBForegrounds.dimensionless_freq_vars(ν, ν0, T)
        @test x ≈ r * x0  # Should be exact mathematical relationship

        # Test x0 calculation: x0 = Ghz_Kelvin * ν0 / T
        expected_x0 = CMBForegrounds.Ghz_Kelvin * ν0 / T
        @test x0 ≈ expected_x0
    end

    @testset "Scaling Properties" begin
        # If we double the frequency, r should double
        ν, ν0, T = 100.0, 50.0, 2.7
        r1, x1, x0_1 = CMBForegrounds.dimensionless_freq_vars(ν, ν0, T)
        r2, x2, x0_2 = CMBForegrounds.dimensionless_freq_vars(2 * ν, ν0, T)

        @test r2 ≈ 2 * r1  # Frequency ratio scales linearly with ν
        @test x2 ≈ 2 * x1  # x scales linearly with ν (since x = r * x0)
        @test x0_2 ≈ x0_1  # x0 shouldn't change when we change ν

        # If we double the reference frequency, r should halve but x0 should double
        r3, x3, x0_3 = CMBForegrounds.dimensionless_freq_vars(ν, 2 * ν0, T)
        @test r3 ≈ r1 / 2    # r = ν/(2*ν0) = (ν/ν0)/2
        @test x0_3 ≈ 2 * x0_1  # x0 = Ghz_Kelvin * (2*ν0) / T
        @test x3 ≈ x1        # x = (r/2) * (2*x0) = r * x0 (unchanged)

        # If we double the temperature, x0 and x should halve
        r4, x4, x0_4 = CMBForegrounds.dimensionless_freq_vars(ν, ν0, 2 * T)
        @test r4 ≈ r1        # r = ν/ν0 (unchanged)
        @test x0_4 ≈ x0_1 / 2  # x0 = Ghz_Kelvin * ν0 / (2*T)
        @test x4 ≈ x1 / 2    # x = r * (x0/2)
    end

    @testset "Type Stability" begin
        # Test with Float64
        r, x, x0 = CMBForegrounds.dimensionless_freq_vars(100.0, 50.0, 2.7)
        @test r isa Float64
        @test x isa Float64
        @test x0 isa Float64

        # Test with Int (should promote to Float64)
        r, x, x0 = CMBForegrounds.dimensionless_freq_vars(100, 50, 3)
        @test r isa Float64  # promote() should convert to Float64
        @test x isa Float64
        @test x0 isa Float64

        # Test with mixed types
        r, x, x0 = CMBForegrounds.dimensionless_freq_vars(100, 50.0, 2.7)
        @test r isa Float64
        @test x isa Float64
        @test x0 isa Float64
    end

    @testset "Edge Cases" begin
        # Test with ν = ν0 (should give r = 1)
        r, x, x0 = CMBForegrounds.dimensionless_freq_vars(143.0, 143.0, 2.7)
        @test r ≈ 1.0
        @test x ≈ x0

        # Test with very small values
        r, x, x0 = CMBForegrounds.dimensionless_freq_vars(0.1, 0.05, 0.001)
        @test r ≈ 2.0  # Should still work
        @test all(isfinite.([r, x, x0]))

        # Test with very large values
        r, x, x0 = CMBForegrounds.dimensionless_freq_vars(1e6, 1e5, 1e3)
        @test r ≈ 10.0
        @test all(isfinite.([r, x, x0]))
    end

    @testset "Physical Consistency" begin
        # Test with realistic CMB values
        T_CMB = CMBForegrounds.T_CMB  # ~2.725 K

        # Planck frequencies
        r_100, x_100, x0_143 = CMBForegrounds.dimensionless_freq_vars(100.0, 143.0, T_CMB)
        r_143, x_143, _ = CMBForegrounds.dimensionless_freq_vars(143.0, 143.0, T_CMB)
        r_217, x_217, _ = CMBForegrounds.dimensionless_freq_vars(217.0, 143.0, T_CMB)

        # Physical expectations
        @test r_100 < r_143 < r_217  # Frequency ratios should increase
        @test r_143 ≈ 1.0           # Reference frequency
        @test x_100 < x_143 < x_217  # Dimensionless frequencies should increase

        # Check that x values are in reasonable range for CMB physics
        @test 1.0 < x_100 < 3.0  # Typical range for CMB frequencies
        @test 2.0 < x_143 < 4.0
        @test 3.0 < x_217 < 5.0
    end

    @testset "Consistency with Constants" begin
        # Test that internal calculation matches expected physics constants
        ν0, T = 100.0, 2.725
        _, _, x0 = CMBForegrounds.dimensionless_freq_vars(100.0, ν0, T)

        # Compare with manual calculation using package constants
        expected_x0 = CMBForegrounds.Ghz_Kelvin * ν0 / T
        @test x0 ≈ expected_x0
    end
end
