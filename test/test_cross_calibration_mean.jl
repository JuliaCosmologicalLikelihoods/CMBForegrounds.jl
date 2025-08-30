"""
Unit tests for cross_calibration_mean function

Tests the cross-calibration mean function that computes:
result = (cal1 * cal2 + cal3 * cal4) / 2

This is a utility function for combining calibration factors from different frequency maps.
"""

@testset "cross_calibration_mean() Unit Tests" begin

    @testset "Basic Functionality" begin
        # Test with typical calibration factors
        cal1, cal2, cal3, cal4 = 0.95, 1.02, 1.01, 0.98

        result = CMBForegrounds.cross_calibration_mean(cal1, cal2, cal3, cal4)

        # Basic output tests
        @test result isa Real
        @test isfinite(result)
        @test typeof(result) <: AbstractFloat

        # Should match the mathematical formula
        expected = (cal1 * cal2 + cal3 * cal4) / 2
        @test result ≈ expected
    end

    @testset "Mathematical Formula Verification" begin
        # Test the exact formula: (cal1 * cal2 + cal3 * cal4) / 2

        # Simple integer case
        result_int = CMBForegrounds.cross_calibration_mean(2, 3, 4, 5)
        expected_int = (2 * 3 + 4 * 5) / 2  # = (6 + 20) / 2 = 13
        @test result_int == expected_int
        @test result_int == 13.0

        # Decimal case
        result_dec = CMBForegrounds.cross_calibration_mean(1.1, 2.2, 3.3, 4.4)
        expected_dec = (1.1 * 2.2 + 3.3 * 4.4) / 2
        @test result_dec ≈ expected_dec

        # Mixed types
        result_mixed = CMBForegrounds.cross_calibration_mean(1, 2.5, 3.0, 4)
        expected_mixed = (1 * 2.5 + 3.0 * 4) / 2  # = (2.5 + 12.0) / 2 = 7.25
        @test result_mixed ≈ expected_mixed
        @test result_mixed == 7.25
    end

    @testset "Mathematical Properties" begin
        # Test mathematical properties of the function

        # Linearity in scaling
        cal1, cal2, cal3, cal4 = 1.0, 1.1, 0.9, 1.2
        scale = 2.0

        result_base = CMBForegrounds.cross_calibration_mean(cal1, cal2, cal3, cal4)
        result_scaled = CMBForegrounds.cross_calibration_mean(scale * cal1, cal2, cal3, cal4)

        # Scaling one factor should affect result proportionally
        expected_scaling = (scale * cal1 * cal2 + cal3 * cal4) / 2
        @test result_scaled ≈ expected_scaling

        # Symmetry properties - swapping pairs should give same result
        result_original = CMBForegrounds.cross_calibration_mean(cal1, cal2, cal3, cal4)
        result_swapped = CMBForegrounds.cross_calibration_mean(cal3, cal4, cal1, cal2)
        @test result_original ≈ result_swapped

        # Swapping within pairs should give same result (multiplication is commutative)
        result_pair_swap = CMBForegrounds.cross_calibration_mean(cal2, cal1, cal4, cal3)
        @test result_original ≈ result_pair_swap
    end

    @testset "Type Stability and Promotion" begin
        # Test with different input types

        # All Float64
        result_float = CMBForegrounds.cross_calibration_mean(1.0, 2.0, 3.0, 4.0)
        @test typeof(result_float) == Float64

        # All integers
        result_int = CMBForegrounds.cross_calibration_mean(1, 2, 3, 4)
        @test typeof(result_int) == Float64  # Division by 2 promotes to Float64

        # Mixed integer and float
        result_mixed = CMBForegrounds.cross_calibration_mean(1, 2.0, 3, 4.0)
        @test typeof(result_mixed) == Float64

        # Should all give the same numerical result for these inputs
        expected_value = (1 * 2 + 3 * 4) / 2  # = 7.0
        @test result_float == expected_value
        @test result_int == expected_value
        @test result_mixed == expected_value

        # Test type promotion with different number types
        result_big = CMBForegrounds.cross_calibration_mean(1, 2, 3, big(4))
        @test typeof(result_big) == BigFloat
    end

    @testset "Physical Realism - Calibration Factors" begin
        # Test with realistic calibration factors from CMB experiments

        # Typical calibration factors are close to 1.0 (few percent deviations)
        realistic_factors = [
            (0.995, 1.003, 1.001, 0.998),  # Very precise calibration
            (0.95, 1.05, 0.98, 1.02),     # Typical calibration uncertainty
            (0.9, 1.1, 0.95, 1.05),       # Larger calibration uncertainty
        ]

        for (cal1, cal2, cal3, cal4) in realistic_factors
            result = CMBForegrounds.cross_calibration_mean(cal1, cal2, cal3, cal4)

            # Result should be close to 1.0 for typical calibrations
            @test abs(result - 1.0) < 0.2  # Within 20% of perfect calibration
            @test result > 0.0  # Should be positive
            @test isfinite(result)

            # Should equal the mathematical formula
            expected = (cal1 * cal2 + cal3 * cal4) / 2
            @test result ≈ expected
        end
    end

    @testset "Edge Cases" begin
        # Test with extreme but mathematically valid inputs

        # All zeros
        result_zeros = CMBForegrounds.cross_calibration_mean(0.0, 0.0, 0.0, 0.0)
        @test result_zeros == 0.0

        # One pair zero
        result_one_zero = CMBForegrounds.cross_calibration_mean(0.0, 1.0, 2.0, 3.0)
        @test result_one_zero == (0.0 * 1.0 + 2.0 * 3.0) / 2
        @test result_one_zero == 3.0

        # Negative calibration factors (mathematically valid but physically unusual)
        result_negative = CMBForegrounds.cross_calibration_mean(-1.0, 2.0, 3.0, 4.0)
        expected_negative = (-1.0 * 2.0 + 3.0 * 4.0) / 2  # = (-2 + 12) / 2 = 5
        @test result_negative ≈ expected_negative
        @test result_negative == 5.0

        # Very large values
        result_large = CMBForegrounds.cross_calibration_mean(1e6, 1e6, 1e6, 1e6)
        expected_large = (1e6 * 1e6 + 1e6 * 1e6) / 2  # = 1e12
        @test result_large ≈ expected_large
        @test isfinite(result_large)

        # Very small values
        result_small = CMBForegrounds.cross_calibration_mean(1e-6, 1e-6, 1e-6, 1e-6)
        expected_small = (1e-6 * 1e-6 + 1e-6 * 1e-6) / 2  # = 1e-12
        @test result_small ≈ expected_small
        @test result_small > 0.0
    end

    @testset "Numerical Precision" begin
        # Test numerical precision with challenging values

        # High precision constants
        π_factor = π
        e_factor = ℯ
        sqrt2_factor = sqrt(2)
        sqrt3_factor = sqrt(3)

        result_precise = CMBForegrounds.cross_calibration_mean(π_factor, e_factor, sqrt2_factor, sqrt3_factor)
        expected_precise = (π * ℯ + sqrt(2) * sqrt(3)) / 2
        @test result_precise ≈ expected_precise

        # Very close values (testing subtraction accuracy)
        base = 1.0000000001
        cal1, cal2 = base, base
        cal3, cal4 = base + 1e-10, base - 1e-10

        result_close = CMBForegrounds.cross_calibration_mean(cal1, cal2, cal3, cal4)
        expected_close = (base^2 + (base + 1e-10) * (base - 1e-10)) / 2
        # = (base^2 + base^2 - (1e-10)^2) / 2 ≈ base^2 (since (1e-10)^2 is tiny)
        @test result_close ≈ expected_close
        @test abs(result_close - base^2) < 1e-15  # Should be very close to base^2
    end

    @testset "Function Properties" begin
        # Test that the function behaves as expected mathematically

        # Identity-like behavior: if all factors are 1, result should be 1
        result_ones = CMBForegrounds.cross_calibration_mean(1.0, 1.0, 1.0, 1.0)
        @test result_ones == 1.0

        # If one product is zero, result is half the other product
        result_half = CMBForegrounds.cross_calibration_mean(0.0, 5.0, 3.0, 4.0)
        @test result_half == (3.0 * 4.0) / 2
        @test result_half == 6.0

        # Distributivity test: function should work with factored forms
        a, b, c, d = 2.0, 3.0, 4.0, 5.0
        result_direct = CMBForegrounds.cross_calibration_mean(a, b, c, d)

        # This is just the same calculation, but verify the arithmetic
        expected_arithmetic = (a * b + c * d) / 2
        @test result_direct ≈ expected_arithmetic

        # Range test: result should be between min and max of the two products
        prod1 = a * b
        prod2 = c * d
        min_prod = min(prod1, prod2)
        max_prod = max(prod1, prod2)

        @test min_prod <= result_direct <= max_prod  # Result is the average, so between the two
    end

    @testset "Inline Function Behavior" begin
        # Test that the @inline function works correctly and efficiently

        # The function should still work normally despite @inline annotation
        cal_vals = [(1.0, 2.0, 3.0, 4.0), (0.5, 1.5, 2.5, 3.5), (10, 20, 30, 40)]

        for (c1, c2, c3, c4) in cal_vals
            result = CMBForegrounds.cross_calibration_mean(c1, c2, c3, c4)
            expected = (c1 * c2 + c3 * c4) / 2
            @test result ≈ expected
        end
    end

    @testset "Calibration Use Cases" begin
        # Test realistic use cases for cross-calibration

        # Case 1: Two frequency maps with cross-calibration
        # Map 1 at freq A with cal factor 0.98, Map 2 at freq B with cal factor 1.02
        # Map 1 at freq C with cal factor 1.01, Map 2 at freq D with cal factor 0.99
        cal_A, cal_B = 0.98, 1.02
        cal_C, cal_D = 1.01, 0.99

        cross_cal = CMBForegrounds.cross_calibration_mean(cal_A, cal_B, cal_C, cal_D)

        # Should give reasonable calibration factor close to 1
        @test abs(cross_cal - 1.0) < 0.1
        @test cross_cal > 0.8 && cross_cal < 1.2  # Reasonable range

        # Manual calculation
        expected_cross = (cal_A * cal_B + cal_C * cal_D) / 2
        @test cross_cal ≈ expected_cross

        # Case 2: Perfect calibration should give 1
        perfect_cal = CMBForegrounds.cross_calibration_mean(1.0, 1.0, 1.0, 1.0)
        @test perfect_cal == 1.0

        # Case 3: Systematic bias
        bias = 0.95  # 5% systematic underestimation
        biased_cal = CMBForegrounds.cross_calibration_mean(bias, bias, bias, bias)
        expected_biased = bias^2  # Since (bias*bias + bias*bias)/2 = bias^2
        @test biased_cal ≈ expected_biased
    end

    @testset "Comparison with Simple Average" begin
        # Test how this differs from simple averaging

        cal1, cal2, cal3, cal4 = 0.9, 1.1, 1.2, 0.8

        # Cross-calibration mean (our function)
        cross_mean = CMBForegrounds.cross_calibration_mean(cal1, cal2, cal3, cal4)

        # Simple arithmetic mean of all factors
        simple_mean = (cal1 + cal2 + cal3 + cal4) / 4

        # They should generally be different (unless special case)
        # Cross mean = (0.9*1.1 + 1.2*0.8)/2 = (0.99 + 0.96)/2 = 0.975
        # Simple mean = (0.9 + 1.1 + 1.2 + 0.8)/4 = 4.0/4 = 1.0
        @test cross_mean != simple_mean

        # Verify the calculations
        @test cross_mean ≈ (0.9 * 1.1 + 1.2 * 0.8) / 2
        @test simple_mean ≈ 1.0

        # But both should be reasonable calibration values
        @test 0.5 < cross_mean < 1.5
        @test 0.5 < simple_mean < 1.5
    end

    @testset "Error Conditions and Robustness" begin
        # Test robustness to various conditions

        # Function should handle all finite real numbers
        test_cases = [
            (1.0, 2.0, 3.0, 4.0),      # Normal positive
            (-1.0, 2.0, 3.0, 4.0),     # One negative
            (-1.0, -2.0, 3.0, 4.0),    # Two negative
            (-1.0, -2.0, -3.0, -4.0),  # All negative
            (0.1, 10.0, 0.2, 5.0),     # Wide range
        ]

        for (c1, c2, c3, c4) in test_cases
            result = CMBForegrounds.cross_calibration_mean(c1, c2, c3, c4)
            expected = (c1 * c2 + c3 * c4) / 2

            @test isfinite(result)
            @test result ≈ expected
        end
    end
end
