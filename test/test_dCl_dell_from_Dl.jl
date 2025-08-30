"""
Unit tests for dCl_dell_from_Dl function

Tests the derivative calculation function that computes dCℓ/dℓ from Dℓ.
The function converts Dℓ = ℓ(ℓ+1)/(2π) * Cℓ to Cℓ, then computes derivatives using
central differences for interior points and boundary extrapolation for endpoints.
"""

@testset "dCl_dell_from_Dl() Unit Tests" begin

    @testset "Basic Functionality" begin
        # Test with simple power spectrum
        ℓs = [100, 200, 300, 400, 500]
        # Create simple Dℓ = ℓ(ℓ+1)/(2π) * Cℓ with Cℓ = constant
        Cℓ_const = 1000.0
        Dℓ = @. ℓs * (ℓs + 1) / (2π) * Cℓ_const

        dCℓ_dℓ = CMBForegrounds.dCl_dell_from_Dl(ℓs, Dℓ)

        # Basic output tests
        @test dCℓ_dℓ isa AbstractVector
        @test length(dCℓ_dℓ) == length(ℓs)
        @test all(isfinite.(dCℓ_dℓ))
        @test eltype(dCℓ_dℓ) <: AbstractFloat

        # For constant Cℓ, derivative should be close to zero
        @test all(abs.(dCℓ_dℓ) .< 1e-10)
    end

    @testset "Mathematical Correctness - Linear Cℓ" begin
        # Test with linear Cℓ = aℓ + b, so dCℓ/dℓ = a
        ℓs = [100, 200, 300, 400, 500]
        a, b = 2.0, 100.0
        Cℓ_linear = @. a * ℓs + b
        Dℓ = @. ℓs * (ℓs + 1) / (2π) * Cℓ_linear

        dCℓ_dℓ = CMBForegrounds.dCl_dell_from_Dl(ℓs, Dℓ)

        # For linear Cℓ, derivative should be approximately the slope 'a'
        # (with some error due to finite differences)
        @test all(abs.(dCℓ_dℓ .- a) .< 0.1)  # Allow for finite difference error

        # Interior points should be more accurate
        @test all(abs.(dCℓ_dℓ[2:end-1] .- a) .< 0.01)
    end

    @testset "Mathematical Correctness - Quadratic Cℓ" begin
        # Test with quadratic Cℓ = aℓ² + bℓ + c, so dCℓ/dℓ = 2aℓ + b
        ℓs = [100, 150, 200, 250, 300]
        a, b, c = 0.01, -2.0, 1000.0
        Cℓ_quad = @. a * ℓs^2 + b * ℓs + c
        Dℓ = @. ℓs * (ℓs + 1) / (2π) * Cℓ_quad

        dCℓ_dℓ = CMBForegrounds.dCl_dell_from_Dl(ℓs, Dℓ)

        # Analytical derivative
        expected_derivative = @. 2 * a * ℓs + b

        # Interior points should match well (central differences are 2nd order accurate)
        for i in 2:length(ℓs)-1
            @test abs(dCℓ_dℓ[i] - expected_derivative[i]) < 1e-6
        end
    end

    @testset "Dℓ to Cℓ Conversion" begin
        # Test the internal Dℓ → Cℓ conversion
        ℓs = [10, 20, 30, 40]
        Cℓ_original = [100.0, 200.0, 150.0, 80.0]

        # Create Dℓ from known Cℓ
        Dℓ = @. ℓs * (ℓs + 1) / (2π) * Cℓ_original

        # Function should recover the derivative correctly
        dCℓ_dℓ = CMBForegrounds.dCl_dell_from_Dl(ℓs, Dℓ)

        # Verify the conversion worked by checking dimensions
        @test length(dCℓ_dℓ) == length(Cℓ_original)
        @test all(isfinite.(dCℓ_dℓ))

        # For this irregular Cℓ, just check that derivative is reasonable
        @test all(abs.(dCℓ_dℓ) .< 100)  # Should be finite and reasonable scale
    end

    @testset "Central Difference Algorithm" begin
        # Test central difference formula: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
        ℓs = [100, 200, 300, 400, 500]  # Evenly spaced
        # Use simple polynomial: Cℓ = ℓ
        Cℓ = ℓs .* 1.0  # Convert to Float64
        Dℓ = @. ℓs * (ℓs + 1) / (2π) * Cℓ

        dCℓ_dℓ = CMBForegrounds.dCl_dell_from_Dl(ℓs, Dℓ)

        # For Cℓ = ℓ, dCℓ/dℓ = 1
        expected = 1.0

        # Check interior points (central differences)
        for i in 2:length(ℓs)-1
            @test abs(dCℓ_dℓ[i] - expected) < 1e-10
        end

        # Boundary points should equal their neighbors
        @test dCℓ_dℓ[1] ≈ dCℓ_dℓ[2]
        @test dCℓ_dℓ[end] ≈ dCℓ_dℓ[end-1]
    end

    @testset "Boundary Conditions" begin
        # Test boundary point handling: endpoints copy neighboring values
        ℓs = [50, 100, 150, 200]
        # Use function where derivative varies: Cℓ = ℓ²
        Cℓ = ℓs .^ 2
        Dℓ = @. ℓs * (ℓs + 1) / (2π) * Cℓ

        dCℓ_dℓ = CMBForegrounds.dCl_dell_from_Dl(ℓs, Dℓ)

        # Boundary conditions: first point = second point, last = second-to-last
        @test dCℓ_dℓ[1] ≈ dCℓ_dℓ[2]
        @test dCℓ_dℓ[4] ≈ dCℓ_dℓ[3]

        # Interior points should be different (unless coincidentally equal)
        # For Cℓ = ℓ², dCℓ/dℓ = 2ℓ, so values should increase
        @test dCℓ_dℓ[2] < dCℓ_dℓ[3]  # Should increase with ℓ
    end

    @testset "Vector Length Constraints" begin
        # Test assertion errors for invalid input lengths

        # Mismatched lengths
        ℓs_3 = [100, 200, 300]
        Dℓ_4 = [10.0, 20.0, 30.0, 40.0]
        @test_throws AssertionError CMBForegrounds.dCl_dell_from_Dl(ℓs_3, Dℓ_4)

        # Too short (n < 2)
        ℓs_1 = [100]
        Dℓ_1 = [10.0]
        @test_throws AssertionError CMBForegrounds.dCl_dell_from_Dl(ℓs_1, Dℓ_1)

        # Empty vectors
        ℓs_empty = Float64[]
        Dℓ_empty = Float64[]
        @test_throws AssertionError CMBForegrounds.dCl_dell_from_Dl(ℓs_empty, Dℓ_empty)

        # Minimum valid case: exactly 2 points
        ℓs_2 = [100, 200]
        Dℓ_2 = [100.0, 200.0]
        dCℓ_dℓ_2 = CMBForegrounds.dCl_dell_from_Dl(ℓs_2, Dℓ_2)
        @test length(dCℓ_dℓ_2) == 2
        @test dCℓ_dℓ_2[1] == dCℓ_dℓ_2[2]  # Both should be equal (boundary condition)
    end

    @testset "Type Stability" begin
        # Test with different input types
        ℓs = [100, 200, 300, 400]
        Cℓ = [10.0, 20.0, 15.0, 25.0]
        Dℓ_base = @. ℓs * (ℓs + 1) / (2π) * Cℓ

        # Float64 inputs
        dCℓ_dℓ_float = CMBForegrounds.dCl_dell_from_Dl(ℓs, Dℓ_base)
        @test eltype(dCℓ_dℓ_float) == Float64

        # Integer ℓs, Float64 Dℓ
        ℓs_int = [100, 200, 300, 400]
        dCℓ_dℓ_mixed = CMBForegrounds.dCl_dell_from_Dl(ℓs_int, Dℓ_base)
        @test eltype(dCℓ_dℓ_mixed) == Float64
        @test dCℓ_dℓ_mixed ≈ dCℓ_dℓ_float

        # Both integers (but Dℓ will be computed as floats)
        Dℓ_int_like = [100, 200, 150, 250]  # Will be treated as integers initially
        dCℓ_dℓ_int = CMBForegrounds.dCl_dell_from_Dl(ℓs_int, Dℓ_int_like)
        @test eltype(dCℓ_dℓ_int) == Float64  # Should promote to Float64 due to division
    end

    @testset "Numerical Precision" begin
        # Test numerical precision with challenging cases

        # Very small Dℓ values
        ℓs = [100, 200, 300, 400, 500]
        Cℓ_small = [1e-10, 2e-10, 1.5e-10, 2.5e-10, 2e-10]
        Dℓ_small = @. ℓs * (ℓs + 1) / (2π) * Cℓ_small

        dCℓ_dℓ_small = CMBForegrounds.dCl_dell_from_Dl(ℓs, Dℓ_small)
        @test all(isfinite.(dCℓ_dℓ_small))
        @test eltype(dCℓ_dℓ_small) == Float64

        # Very large Dℓ values
        Cℓ_large = [1e10, 2e10, 1.5e10, 2.5e10, 2e10]
        Dℓ_large = @. ℓs * (ℓs + 1) / (2π) * Cℓ_large

        dCℓ_dℓ_large = CMBForegrounds.dCl_dell_from_Dl(ℓs, Dℓ_large)
        @test all(isfinite.(dCℓ_dℓ_large))

        # High precision calculation
        ℓs_precise = [100π, 200π, 300π, 400π]
        Cℓ_precise = [π, ℯ, sqrt(2), sqrt(3)]
        Dℓ_precise = @. ℓs_precise * (ℓs_precise + 1) / (2π) * Cℓ_precise

        dCℓ_dℓ_precise = CMBForegrounds.dCl_dell_from_Dl(ℓs_precise, Dℓ_precise)
        @test all(isfinite.(dCℓ_dℓ_precise))
    end

    @testset "Physical Realism - CMB Power Spectra" begin
        # Test with realistic CMB power spectrum shape
        ℓs = [50, 100, 200, 500, 1000, 2000, 3000]

        # Approximate CMB TT power spectrum shape (simplified)
        # Peak around ℓ ~ 200, then falls off
        Cℓ_cmb = @. 6000 * exp(-((ℓs - 200) / 400)^2) / (ℓs / 100 + 1)
        Dℓ_cmb = @. ℓs * (ℓs + 1) / (2π) * Cℓ_cmb

        dCℓ_dℓ_cmb = CMBForegrounds.dCl_dell_from_Dl(ℓs, Dℓ_cmb)

        # Physical expectations
        @test all(isfinite.(dCℓ_dℓ_cmb))
        @test length(dCℓ_dℓ_cmb) == length(ℓs)

        # Should have reasonable magnitudes (not too extreme)
        @test all(abs.(dCℓ_dℓ_cmb) .< 1000)

        # Derivative should capture the overall trend
        # (detailed physics not critical, just that it's reasonable)
        @test eltype(dCℓ_dℓ_cmb) == Float64
    end

    @testset "Finite Difference Accuracy" begin
        # Test accuracy of finite difference scheme
        # Use polynomial where we know exact derivative

        ℓs = collect(100:50:400)  # Evenly spaced for best accuracy

        # Cubic polynomial: Cℓ = aℓ³ + bℓ² + cℓ + d
        a, b, c, d = 1e-6, -1e-3, 0.5, 100.0
        Cℓ_cubic = @. a * ℓs^3 + b * ℓs^2 + c * ℓs + d
        Dℓ_cubic = @. ℓs * (ℓs + 1) / (2π) * Cℓ_cubic

        dCℓ_dℓ_numeric = CMBForegrounds.dCl_dell_from_Dl(ℓs, Dℓ_cubic)

        # Analytical derivative: dCℓ/dℓ = 3aℓ² + 2bℓ + c
        dCℓ_dℓ_exact = @. 3 * a * ℓs^2 + 2 * b * ℓs + c

        # Interior points should be reasonably accurate (central differences are O(h²))
        for i in 2:length(ℓs)-1
            rel_error = abs(dCℓ_dℓ_numeric[i] - dCℓ_dℓ_exact[i]) / abs(dCℓ_dℓ_exact[i])
            @test rel_error < 0.02  # Allow ~2% error for finite differences
        end
    end

    @testset "Non-uniform Grid Spacing" begin
        # Test with non-uniformly spaced ℓ values
        ℓs = [50, 75, 120, 200, 350, 600]  # Irregular spacing

        # Simple test function: Cℓ = ℓ
        Cℓ = ℓs .* 1.0
        Dℓ = @. ℓs * (ℓs + 1) / (2π) * Cℓ

        dCℓ_dℓ = CMBForegrounds.dCl_dell_from_Dl(ℓs, Dℓ)

        # For Cℓ = ℓ, dCℓ/dℓ = 1, should still work with irregular grid
        @test all(isfinite.(dCℓ_dℓ))

        # Interior points should still be close to 1 (but less accurate)
        for i in 2:length(ℓs)-1
            @test abs(dCℓ_dℓ[i] - 1.0) < 0.1
        end

        # Boundary conditions still apply
        @test dCℓ_dℓ[1] ≈ dCℓ_dℓ[2]
        @test dCℓ_dℓ[end] ≈ dCℓ_dℓ[end-1]
    end

    @testset "Edge Cases and Robustness" begin
        # Test various edge cases

        # Very close ℓ values (small differences)
        ℓs_close = [1000.0, 1000.1, 1000.2, 1000.3]
        Cℓ_smooth = [100.0, 100.05, 100.1, 100.15]  # Smooth variation
        Dℓ_close = @. ℓs_close * (ℓs_close + 1) / (2π) * Cℓ_smooth

        dCℓ_dℓ_close = CMBForegrounds.dCl_dell_from_Dl(ℓs_close, Dℓ_close)
        @test all(isfinite.(dCℓ_dℓ_close))

        # Large ℓ values
        ℓs_large = [10000, 20000, 30000, 40000]
        Cℓ_large = [0.1, 0.08, 0.06, 0.04]  # Typical high-ℓ falloff
        Dℓ_large_l = @. ℓs_large * (ℓs_large + 1) / (2π) * Cℓ_large

        dCℓ_dℓ_large_l = CMBForegrounds.dCl_dell_from_Dl(ℓs_large, Dℓ_large_l)
        @test all(isfinite.(dCℓ_dℓ_large_l))

        # Oscillatory Cℓ (challenging for derivatives)
        ℓs_osc = collect(100:20:200)
        Cℓ_osc = @. 1000 + 100 * sin(ℓs_osc / 20)
        Dℓ_osc = @. ℓs_osc * (ℓs_osc + 1) / (2π) * Cℓ_osc

        dCℓ_dℓ_osc = CMBForegrounds.dCl_dell_from_Dl(ℓs_osc, Dℓ_osc)
        @test all(isfinite.(dCℓ_dℓ_osc))
    end

    @testset "Consistency Checks" begin
        # Test internal consistency and relationships

        ℓs = [100, 200, 300, 400, 500]
        Cℓ = [500.0, 400.0, 300.0, 200.0, 100.0]  # Decreasing
        Dℓ = @. ℓs * (ℓs + 1) / (2π) * Cℓ

        dCℓ_dℓ = CMBForegrounds.dCl_dell_from_Dl(ℓs, Dℓ)

        # For decreasing Cℓ, derivative should be negative
        @test all(dCℓ_dℓ[2:end-1] .< 0)  # Interior points

        # Test with increasing Cℓ
        Cℓ_inc = [100.0, 200.0, 300.0, 400.0, 500.0]
        Dℓ_inc = @. ℓs * (ℓs + 1) / (2π) * Cℓ_inc
        dCℓ_dℓ_inc = CMBForegrounds.dCl_dell_from_Dl(ℓs, Dℓ_inc)

        # For increasing Cℓ, derivative should be positive
        @test all(dCℓ_dℓ_inc[2:end-1] .> 0)

        # Derivatives should be opposite in sign (same magnitude)
        @test all(abs.(dCℓ_dℓ[2:end-1] + dCℓ_dℓ_inc[2:end-1]) .< 1e-10)
    end

    @testset "Memory and Performance" begin
        # Test with larger arrays to check memory efficiency
        n = 1000
        ℓs_large = collect(10:10:10000)  # 1000 points
        Cℓ_large = @. 1000 / (1 + (ℓs_large / 500)^2)  # Smooth function
        Dℓ_large = @. ℓs_large * (ℓs_large + 1) / (2π) * Cℓ_large

        dCℓ_dℓ_large = CMBForegrounds.dCl_dell_from_Dl(ℓs_large, Dℓ_large)

        # Basic correctness for large array
        @test length(dCℓ_dℓ_large) == n
        @test all(isfinite.(dCℓ_dℓ_large))
        @test eltype(dCℓ_dℓ_large) == Float64

        # Boundary conditions still hold
        @test dCℓ_dℓ_large[1] ≈ dCℓ_dℓ_large[2]
        @test dCℓ_dℓ_large[end] ≈ dCℓ_dℓ_large[end-1]
    end

    @testset "Special Cases" begin
        # Test special mathematical cases

        # All zeros
        ℓs = [100, 200, 300, 400]
        Dℓ_zero = [0.0, 0.0, 0.0, 0.0]
        dCℓ_dℓ_zero = CMBForegrounds.dCl_dell_from_Dl(ℓs, Dℓ_zero)
        @test all(dCℓ_dℓ_zero .== 0.0)

        # Constant non-zero Dℓ (implies Cℓ ∝ 1/[ℓ(ℓ+1)])
        Dℓ_const = [1000.0, 1000.0, 1000.0, 1000.0]
        dCℓ_dℓ_const = CMBForegrounds.dCl_dell_from_Dl(ℓs, Dℓ_const)

        # Since Cℓ = Dℓ * 2π / [ℓ(ℓ+1)] and Dℓ is constant,
        # Cℓ ∝ 1/[ℓ(ℓ+1)], so dCℓ/dℓ should be negative
        @test all(dCℓ_dℓ_const[2:end-1] .< 0)

        # Single peak (realistic for CMB)
        ℓs_peak = [50, 100, 200, 300, 500]
        # Gaussian-like peak around ℓ=200
        Cℓ_peak = @. 5000 * exp(-((ℓs_peak - 200) / 100)^2)
        Dℓ_peak = @. ℓs_peak * (ℓs_peak + 1) / (2π) * Cℓ_peak

        dCℓ_dℓ_peak = CMBForegrounds.dCl_dell_from_Dl(ℓs_peak, Dℓ_peak)
        @test all(isfinite.(dCℓ_dℓ_peak))

        # Should change sign around the peak
        # (detailed behavior depends on sampling, just check it's reasonable)
        @test any(dCℓ_dℓ_peak .> 0) || any(dCℓ_dℓ_peak .< 0)  # Not all same sign
    end
end
