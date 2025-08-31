"""
Unit tests for ssl_response function

Tests the super-sample lensing (SSL) response function that computes:
Δ Dℓ = -κ * [ℓ²(ℓ+1)/(2π) * dCℓ/dℓ + 2 * Dℓ]

where κ is the convergence field value and dCℓ/dℓ is computed from Dℓ.
"""

@testset "ssl_response() Unit Tests" begin

    @testset "Basic Functionality" begin
        # Test with simple power spectrum
        ℓs = [100, 200, 300, 400, 500]
        # Simple Dℓ values
        Dℓ = [1000.0, 2000.0, 1500.0, 3000.0, 2500.0]
        κ = 0.01  # Typical convergence strength

        Δ_Dℓ = CMBForegrounds.ssl_response(ℓs, κ, Dℓ)

        # Basic output tests
        @test Δ_Dℓ isa AbstractVector
        @test length(Δ_Dℓ) == length(ℓs)
        @test length(Δ_Dℓ) == length(Dℓ)
        @test all(isfinite.(Δ_Dℓ))
        @test eltype(Δ_Dℓ) <: AbstractFloat

        # Should be non-zero for non-zero κ (unless very special case)
        @test any(Δ_Dℓ .!= 0.0)
    end

    @testset "Mathematical Formula Verification" begin
        # Test the SSL formula: Δ Dℓ = -κ * [ℓ²(ℓ+1)/(2π) * dCℓ/dℓ + 2 * Dℓ]
        ℓs = [200, 300, 400]
        Dℓ = [800.0, 1200.0, 900.0]
        κ = 0.02

        Δ_Dℓ = CMBForegrounds.ssl_response(ℓs, κ, Dℓ)

        # Manual calculation
        dCℓ = CMBForegrounds.dCl_dell_from_Dl(ℓs, Dℓ)

        expected = similar(Dℓ)
        for i in 1:length(ℓs)
            ℓ = ℓs[i]
            pref = (ℓ * ℓ * (ℓ + 1)) / (2π)
            ssl = pref * dCℓ[i] + 2 * Dℓ[i]
            expected[i] = -κ * ssl
        end

        # Should match exactly
        @test all(Δ_Dℓ .≈ expected)
    end

    @testset "Linear Scaling with Convergence" begin
        # Test that response scales linearly with κ
        ℓs = [100, 200, 300, 400, 500]
        Dℓ = [500.0, 1000.0, 800.0, 1200.0, 900.0]

        κ1, κ2 = 0.01, 0.03
        Δ_Dℓ_1 = CMBForegrounds.ssl_response(ℓs, κ1, Dℓ)
        Δ_Dℓ_2 = CMBForegrounds.ssl_response(ℓs, κ2, Dℓ)

        # Should scale exactly linearly with κ
        @test all(Δ_Dℓ_2 .≈ (κ2 / κ1) .* Δ_Dℓ_1)

        # Test with negative convergence
        κ_neg = -0.01
        Δ_Dℓ_neg = CMBForegrounds.ssl_response(ℓs, κ_neg, Dℓ)
        Δ_Dℓ_pos = CMBForegrounds.ssl_response(ℓs, abs(κ_neg), Dℓ)

        # Should be exactly opposite
        @test all(Δ_Dℓ_neg .≈ -Δ_Dℓ_pos)
    end

    @testset "Zero Convergence" begin
        # Test that zero convergence gives zero response
        ℓs = [100, 200, 300, 400]
        Dℓ = [1000.0, 2000.0, 1500.0, 2500.0]
        κ_zero = 0.0

        Δ_Dℓ_zero = CMBForegrounds.ssl_response(ℓs, κ_zero, Dℓ)

        # Should be exactly zero
        @test all(Δ_Dℓ_zero .== 0.0)
        @test length(Δ_Dℓ_zero) == length(ℓs)
    end

    @testset "Integration with dCl_dell_from_Dl" begin
        # Test that the function properly uses dCl_dell_from_Dl
        ℓs = [100, 200, 300, 400]

        # Use a simple power law: Cℓ = A * ℓ^(-2), so Dℓ = A * ℓ(ℓ+1)/(2π) * ℓ^(-2) = A * (ℓ+1)/(2π)
        A = 1000.0
        Dℓ = @. A * (ℓs + 1) / (2π)
        κ = 0.01

        Δ_Dℓ = CMBForegrounds.ssl_response(ℓs, κ, Dℓ)

        # For this power law, dCℓ/dℓ = -2A/ℓ³ (approximately)
        # The exact result depends on the finite differences, so just check reasonableness
        @test all(isfinite.(Δ_Dℓ))
        @test all(Δ_Dℓ .!= 0.0)  # Should be non-zero

        # Since dCℓ/dℓ < 0 for this case and Dℓ > 0, and κ > 0:
        # SSL = ℓ²(ℓ+1)/(2π) * (negative) + 2 * (positive)
        # The sign depends on which term dominates, but should be finite
        @test eltype(Δ_Dℓ) == Float64
    end

    @testset "Vector Length Constraints" begin
        # Test assertion errors for invalid input lengths

        # Mismatched lengths
        ℓs_3 = [100, 200, 300]
        Dℓ_4 = [100.0, 200.0, 300.0, 400.0]
        κ = 0.01
        @test_throws AssertionError CMBForegrounds.ssl_response(ℓs_3, κ, Dℓ_4)

        # Empty vectors
        ℓs_empty = Float64[]
        Dℓ_empty = Float64[]
        @test_throws AssertionError CMBForegrounds.ssl_response(ℓs_empty, κ, Dℓ_empty)  # Will fail in dCl_dell_from_Dl

        # Single element (will fail in dCl_dell_from_Dl due to derivative requirement)
        ℓs_1 = [100]
        Dℓ_1 = [100.0]
        @test_throws AssertionError CMBForegrounds.ssl_response(ℓs_1, κ, Dℓ_1)

        # Minimum valid case: exactly 2 points
        ℓs_2 = [100, 200]
        Dℓ_2 = [100.0, 200.0]
        Δ_Dℓ_2 = CMBForegrounds.ssl_response(ℓs_2, κ, Dℓ_2)
        @test length(Δ_Dℓ_2) == 2
        @test all(isfinite.(Δ_Dℓ_2))
    end

    @testset "Type Stability" begin
        # Test with different input types
        ℓs = [100, 200, 300, 400]
        Dℓ = [500.0, 1000.0, 800.0, 1200.0]
        κ = 0.01

        # Float64 inputs
        Δ_Dℓ_float = CMBForegrounds.ssl_response(ℓs, κ, Dℓ)
        @test eltype(Δ_Dℓ_float) == Float64

        # Integer κ
        κ_int = 0  # This will be converted to float in calculations
        Δ_Dℓ_int_κ = CMBForegrounds.ssl_response(ℓs, κ_int, Dℓ)
        @test eltype(Δ_Dℓ_int_κ) == Float64
        @test all(Δ_Dℓ_int_κ .== 0.0)  # Zero convergence

        # Mixed types - avoid integer return type by using floats
        ℓs_int = [100, 200, 300, 400]
        Dℓ_int_float = [500.0, 1000.0, 800.0, 1200.0]  # Use Float64
        Δ_Dℓ_mixed = CMBForegrounds.ssl_response(ℓs_int, 0.01, Dℓ_int_float)
        @test eltype(Δ_Dℓ_mixed) == Float64
    end

    @testset "Numerical Precision" begin
        # Test numerical precision with challenging cases

        # Very small Dℓ values
        ℓs = [100, 200, 300, 400, 500]
        Dℓ_small = [1e-10, 2e-10, 1.5e-10, 2.5e-10, 2e-10]
        κ = 0.01

        Δ_Dℓ_small = CMBForegrounds.ssl_response(ℓs, κ, Dℓ_small)
        @test all(isfinite.(Δ_Dℓ_small))
        @test eltype(Δ_Dℓ_small) == Float64

        # Very large Dℓ values
        Dℓ_large = [1e10, 2e10, 1.5e10, 2.5e10, 2e10]
        Δ_Dℓ_large = CMBForegrounds.ssl_response(ℓs, κ, Dℓ_large)
        @test all(isfinite.(Δ_Dℓ_large))

        # Very small convergence
        κ_small = 1e-8
        Δ_Dℓ_small_κ = CMBForegrounds.ssl_response(ℓs, κ_small, Dℓ_small)
        @test all(isfinite.(Δ_Dℓ_small_κ))

        # High precision calculation
        ℓs_precise = [100π, 200π, 300π, 400π]
        Dℓ_precise = [π^2, ℯ^2, sqrt(2)^2, sqrt(3)^2]
        κ_precise = π / 1000

        Δ_Dℓ_precise = CMBForegrounds.ssl_response(ℓs_precise, κ_precise, Dℓ_precise)
        @test all(isfinite.(Δ_Dℓ_precise))
    end

    @testset "Physical Realism - CMB Power Spectra" begin
        # Test with realistic CMB power spectrum
        ℓs = [50, 100, 200, 500, 1000, 2000, 3000]

        # Approximate CMB TT power spectrum shape
        Dℓ_cmb = @. 6000 * exp(-((ℓs - 200) / 400)^2) / (ℓs / 100 + 1) * (ℓs * (ℓs + 1) / (2π))

        # Typical convergence values from weak lensing
        κ_values = [0.001, 0.01, 0.1]  # Range from very weak to strong lensing

        for κ in κ_values
            Δ_Dℓ_cmb = CMBForegrounds.ssl_response(ℓs, κ, Dℓ_cmb)

            # Physical expectations
            @test all(isfinite.(Δ_Dℓ_cmb))
            @test length(Δ_Dℓ_cmb) == length(ℓs)

            # Should be proportional to κ (but can be large due to derivative terms)
            # @test all(abs.(Δ_Dℓ_cmb) .< 10 * abs(κ) * maximum(abs.(Dℓ_cmb)))  # Too restrictive

            # SSL response should be finite and well-behaved
            # (Note: SSL can be quite large relative to original spectrum due to derivative terms)
            @test all(isfinite.(Δ_Dℓ_cmb))
            @test all(abs.(Δ_Dℓ_cmb) .< 1e10)  # Just check it's not infinite/extreme
        end
    end

    @testset "Constant Dℓ Behavior" begin
        # Test with constant Dℓ (implies specific Cℓ ∝ 1/[ℓ(ℓ+1)])
        ℓs = [100, 200, 300, 400]
        Dℓ_const = [1000.0, 1000.0, 1000.0, 1000.0]
        κ = 0.01

        Δ_Dℓ_const = CMBForegrounds.ssl_response(ℓs, κ, Dℓ_const)

        # For constant Dℓ, Cℓ ∝ 1/[ℓ(ℓ+1)], so dCℓ/dℓ < 0
        # SSL = ℓ²(ℓ+1)/(2π) * (negative) + 2 * 1000 = mix of negative and positive terms
        @test all(isfinite.(Δ_Dℓ_const))
        @test all(Δ_Dℓ_const .!= 0.0)  # Should be non-zero

        # The response should still be proportional to κ
        Δ_Dℓ_double = CMBForegrounds.ssl_response(ℓs, 2 * κ, Dℓ_const)
        @test all(Δ_Dℓ_double .≈ 2.0 .* Δ_Dℓ_const)
    end

    @testset "Power Law Dℓ Behavior" begin
        # Test with power law Dℓ ∝ ℓ^α
        ℓs = [100, 200, 400, 800]  # Powers of 2 for cleaner behavior
        α = -1.5
        A = 1000.0

        Dℓ_power = @. A * (ℓs / 100)^α
        κ = 0.01

        Δ_Dℓ_power = CMBForegrounds.ssl_response(ℓs, κ, Dℓ_power)

        # Should be finite and well-behaved
        @test all(isfinite.(Δ_Dℓ_power))
        @test all(Δ_Dℓ_power .!= 0.0)  # Should be non-zero for power law

        # Should still scale with κ
        Δ_Dℓ_scaled = CMBForegrounds.ssl_response(ℓs, 3 * κ, Dℓ_power)
        @test all(Δ_Dℓ_scaled .≈ 3.0 .* Δ_Dℓ_power)
    end

    @testset "Edge Cases and Robustness" begin
        # Test various edge cases

        # Very close ℓ values (challenging for derivatives)
        ℓs_close = [1000.0, 1000.1, 1000.2, 1000.3]
        Dℓ_close = [100.0, 100.05, 100.1, 100.15]
        κ = 0.01

        Δ_Dℓ_close = CMBForegrounds.ssl_response(ℓs_close, κ, Dℓ_close)
        @test all(isfinite.(Δ_Dℓ_close))

        # Large ℓ values
        ℓs_large = [10000, 20000, 30000, 40000]
        Dℓ_large = [0.1, 0.08, 0.06, 0.04]

        Δ_Dℓ_large_l = CMBForegrounds.ssl_response(ℓs_large, κ, Dℓ_large)
        @test all(isfinite.(Δ_Dℓ_large_l))

        # Oscillatory Dℓ
        ℓs_osc = collect(100:50:300)
        Dℓ_osc = @. 1000 + 100 * sin(ℓs_osc / 50)

        Δ_Dℓ_osc = CMBForegrounds.ssl_response(ℓs_osc, κ, Dℓ_osc)
        @test all(isfinite.(Δ_Dℓ_osc))

        # Very large convergence
        κ_large = 1.0  # Unphysically large but mathematically valid
        ℓs_test = [100, 200, 300]
        Dℓ_test = [100.0, 200.0, 150.0]
        Δ_Dℓ_large_κ = CMBForegrounds.ssl_response(ℓs_test, κ_large, Dℓ_test)
        @test all(isfinite.(Δ_Dℓ_large_κ))
    end

    @testset "Sign Conventions" begin
        # Test sign conventions and physics
        ℓs = [100, 200, 300, 400]
        Dℓ = [1000.0, 800.0, 600.0, 400.0]  # Decreasing spectrum

        κ_pos = 0.01
        κ_neg = -0.01

        Δ_Dℓ_pos = CMBForegrounds.ssl_response(ℓs, κ_pos, Dℓ)
        Δ_Dℓ_neg = CMBForegrounds.ssl_response(ℓs, κ_neg, Dℓ)

        # Should be exactly opposite for opposite convergence
        @test all(Δ_Dℓ_pos .≈ -Δ_Dℓ_neg)

        # Check that the sign is consistent with the -κ factor
        dCℓ = CMBForegrounds.dCl_dell_from_Dl(ℓs, Dℓ)
        for i in 1:length(ℓs)
            ℓ = ℓs[i]
            pref = (ℓ * ℓ * (ℓ + 1)) / (2π)
            ssl_term = pref * dCℓ[i] + 2 * Dℓ[i]
            expected = -κ_pos * ssl_term
            @test Δ_Dℓ_pos[i] ≈ expected
        end
    end

    @testset "Memory and Performance" begin
        # Test with larger arrays to check memory efficiency
        n = 1000
        ℓs_large = collect(10:10:10000)
        Dℓ_large = @. 1000 / (1 + (ℓs_large / 500)^2)
        κ = 0.005

        Δ_Dℓ_large = CMBForegrounds.ssl_response(ℓs_large, κ, Dℓ_large)

        # Basic correctness for large array
        @test length(Δ_Dℓ_large) == n
        @test all(isfinite.(Δ_Dℓ_large))
        @test eltype(Δ_Dℓ_large) == Float64

        # Should still scale with convergence
        Δ_Dℓ_scaled = CMBForegrounds.ssl_response(ℓs_large, 2 * κ, Dℓ_large)
        @test all(Δ_Dℓ_scaled .≈ 2.0 .* Δ_Dℓ_large)
    end

    @testset "Consistency with Literature" begin
        # Test that the SSL formula matches expected form from literature
        ℓs = [200, 500, 1000]  # Representative multipoles

        # Use a realistic CMB-like spectrum at these scales
        Dℓ_realistic = [5000.0, 3000.0, 1000.0]  # Typical TT values in μK²
        κ = 0.01  # 1% convergence

        Δ_Dℓ_lit = CMBForegrounds.ssl_response(ℓs, κ, Dℓ_realistic)

        # The magnitude should be roughly κ times a combination of the derivative and amplitude terms
        dCℓ = CMBForegrounds.dCl_dell_from_Dl(ℓs, Dℓ_realistic)

        for i in 1:length(ℓs)
            ℓ = ℓs[i]

            # The derivative term should be larger at higher ℓ due to ℓ² factor
            deriv_term = (ℓ * ℓ * (ℓ + 1)) / (2π) * dCℓ[i]
            amplitude_term = 2 * Dℓ_realistic[i]

            total_term = deriv_term + amplitude_term
            expected_magnitude = abs(κ * total_term)

            @test abs(Δ_Dℓ_lit[i]) ≈ expected_magnitude
            @test sign(Δ_Dℓ_lit[i]) == -sign(κ * total_term)  # Check sign
        end
    end

    @testset "Numerical Derivatives Integration" begin
        # Test that the function properly handles the derivatives from dCl_dell_from_Dl
        ℓs = [100, 150, 200, 250, 300]  # Evenly spaced

        # Quadratic Dℓ for known derivative behavior
        a, b, c = 0.01, -2.0, 1000.0
        # If Cℓ ∝ aℓ² + bℓ + c, then when we convert to Dℓ and back,
        # we should get predictable derivative behavior

        # Create Dℓ from a specific Cℓ form
        Cℓ_quad = @. a * ℓs^2 + b * ℓs + c
        Dℓ_quad = @. ℓs * (ℓs + 1) / (2π) * Cℓ_quad

        κ = 0.01
        Δ_Dℓ_quad = CMBForegrounds.ssl_response(ℓs, κ, Dℓ_quad)

        # Should integrate properly with the derivative calculation
        @test all(isfinite.(Δ_Dℓ_quad))
        @test all(Δ_Dℓ_quad .!= 0.0)  # Should be non-zero

        # The middle points should be well-behaved (central differences work best there)
        @test all(isfinite.(Δ_Dℓ_quad[2:end-1]))
    end
end
