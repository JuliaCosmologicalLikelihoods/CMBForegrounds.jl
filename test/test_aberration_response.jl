"""
Unit tests for aberration_response function

Tests the relativistic aberration response function that computes:
Δ Dℓ = -ab_coeff * dCℓ/dℓ * ℓ²(ℓ+1)/(2π)

where ab_coeff is the aberration coefficient related to observer velocity and dCℓ/dℓ is computed from Dℓ.
"""

using Test
using CMBForegrounds

@testset "aberration_response() Unit Tests" begin
    
    @testset "Basic Functionality" begin
        # Test with simple power spectrum
        ℓs = [100, 200, 300, 400, 500]
        # Simple Dℓ values
        Dℓ = [1000.0, 2000.0, 1500.0, 3000.0, 2500.0]
        ab_coeff = 1e-3  # Typical aberration coefficient
        
        Δ_Dℓ = CMBForegrounds.aberration_response(ℓs, ab_coeff, Dℓ)
        
        # Basic output tests
        @test Δ_Dℓ isa AbstractVector
        @test length(Δ_Dℓ) == length(ℓs)
        @test length(Δ_Dℓ) == length(Dℓ)
        @test all(isfinite.(Δ_Dℓ))
        @test eltype(Δ_Dℓ) <: AbstractFloat
        
        # Should be non-zero for non-zero ab_coeff (unless very special case)
        @test any(Δ_Dℓ .!= 0.0)
    end
    
    @testset "Mathematical Formula Verification" begin
        # Test the aberration formula: Δ Dℓ = -ab_coeff * dCℓ/dℓ * ℓ²(ℓ+1)/(2π)
        ℓs = [200, 300, 400]
        Dℓ = [800.0, 1200.0, 900.0]
        ab_coeff = 2e-3
        
        Δ_Dℓ = CMBForegrounds.aberration_response(ℓs, ab_coeff, Dℓ)
        
        # Manual calculation
        dCℓ = CMBForegrounds.dCl_dell_from_Dl(ℓs, Dℓ)
        
        expected = similar(Dℓ)
        for i in 1:length(ℓs)
            ℓ = ℓs[i]
            pref = (ℓ * ℓ * (ℓ + 1)) / (2π)
            expected[i] = -ab_coeff * dCℓ[i] * pref
        end
        
        # Should match exactly
        @test all(Δ_Dℓ .≈ expected)
    end
    
    @testset "Linear Scaling with Aberration Coefficient" begin
        # Test that response scales linearly with ab_coeff
        ℓs = [100, 200, 300, 400, 500]
        Dℓ = [500.0, 1000.0, 800.0, 1200.0, 900.0]
        
        ab1, ab2 = 1e-3, 3e-3
        Δ_Dℓ_1 = CMBForegrounds.aberration_response(ℓs, ab1, Dℓ)
        Δ_Dℓ_2 = CMBForegrounds.aberration_response(ℓs, ab2, Dℓ)
        
        # Should scale exactly linearly with ab_coeff
        @test all(Δ_Dℓ_2 .≈ (ab2/ab1) .* Δ_Dℓ_1)
        
        # Test with negative aberration coefficient
        ab_neg = -1e-3
        Δ_Dℓ_neg = CMBForegrounds.aberration_response(ℓs, ab_neg, Dℓ)
        Δ_Dℓ_pos = CMBForegrounds.aberration_response(ℓs, abs(ab_neg), Dℓ)
        
        # Should be exactly opposite
        @test all(Δ_Dℓ_neg .≈ -Δ_Dℓ_pos)
    end
    
    @testset "Zero Aberration Coefficient" begin
        # Test that zero aberration coefficient gives zero response
        ℓs = [100, 200, 300, 400]
        Dℓ = [1000.0, 2000.0, 1500.0, 2500.0]
        ab_coeff_zero = 0.0
        
        Δ_Dℓ_zero = CMBForegrounds.aberration_response(ℓs, ab_coeff_zero, Dℓ)
        
        # Should be exactly zero
        @test all(Δ_Dℓ_zero .== 0.0)
        @test length(Δ_Dℓ_zero) == length(ℓs)
    end
    
    @testset "Integration with dCl_dell_from_Dl" begin
        # Test that the function properly uses dCl_dell_from_Dl
        ℓs = [100, 200, 300, 400]
        
        # Use a simple power law: Cℓ = A * ℓ^(-2)
        A = 1000.0
        Dℓ = @. A * (ℓs + 1) / (2π)  # Approximate Dℓ for this Cℓ
        ab_coeff = 1e-3
        
        Δ_Dℓ = CMBForegrounds.aberration_response(ℓs, ab_coeff, Dℓ)
        
        # Should integrate properly with the derivative calculation
        @test all(isfinite.(Δ_Dℓ))
        @test all(Δ_Dℓ .!= 0.0)  # Should be non-zero
        
        # Since dCℓ/dℓ < 0 for decreasing power law and ab_coeff > 0:
        # Aberration = -ab_coeff * (negative) * (positive) = positive * positive = positive
        # But this depends on the exact derivative behavior from finite differences
        @test eltype(Δ_Dℓ) == Float64
    end
    
    @testset "Vector Length Constraints" begin
        # Test assertion errors for invalid input lengths
        
        # Mismatched lengths
        ℓs_3 = [100, 200, 300]
        Dℓ_4 = [100.0, 200.0, 300.0, 400.0]
        ab_coeff = 1e-3
        @test_throws AssertionError CMBForegrounds.aberration_response(ℓs_3, ab_coeff, Dℓ_4)
        
        # Empty vectors
        ℓs_empty = Float64[]
        Dℓ_empty = Float64[]
        @test_throws AssertionError CMBForegrounds.aberration_response(ℓs_empty, ab_coeff, Dℓ_empty)  # Will fail in dCl_dell_from_Dl
        
        # Single element (will fail in dCl_dell_from_Dl due to derivative requirement)
        ℓs_1 = [100]
        Dℓ_1 = [100.0]
        @test_throws AssertionError CMBForegrounds.aberration_response(ℓs_1, ab_coeff, Dℓ_1)
        
        # Minimum valid case: exactly 2 points
        ℓs_2 = [100, 200]
        Dℓ_2 = [100.0, 200.0]
        Δ_Dℓ_2 = CMBForegrounds.aberration_response(ℓs_2, ab_coeff, Dℓ_2)
        @test length(Δ_Dℓ_2) == 2
        @test all(isfinite.(Δ_Dℓ_2))
    end
    
    @testset "Type Stability" begin
        # Test with different input types
        ℓs = [100, 200, 300, 400]
        Dℓ = [500.0, 1000.0, 800.0, 1200.0]
        ab_coeff = 1e-3
        
        # Float64 inputs
        Δ_Dℓ_float = CMBForegrounds.aberration_response(ℓs, ab_coeff, Dℓ)
        @test eltype(Δ_Dℓ_float) == Float64
        
        # Integer ab_coeff
        ab_coeff_int = 0  # This will be converted to float in calculations
        Δ_Dℓ_int_ab = CMBForegrounds.aberration_response(ℓs, ab_coeff_int, Dℓ)
        @test eltype(Δ_Dℓ_int_ab) == Float64
        @test all(Δ_Dℓ_int_ab .== 0.0)  # Zero aberration
        
        # Mixed types - avoid integer return type by using floats
        ℓs_int = [100, 200, 300, 400]
        Dℓ_int_float = [500.0, 1000.0, 800.0, 1200.0]  # Use Float64
        Δ_Dℓ_mixed = CMBForegrounds.aberration_response(ℓs_int, 1e-3, Dℓ_int_float)
        @test eltype(Δ_Dℓ_mixed) == Float64
    end
    
    @testset "Numerical Precision" begin
        # Test numerical precision with challenging cases
        
        # Very small Dℓ values
        ℓs = [100, 200, 300, 400, 500]
        Dℓ_small = [1e-10, 2e-10, 1.5e-10, 2.5e-10, 2e-10]
        ab_coeff = 1e-3
        
        Δ_Dℓ_small = CMBForegrounds.aberration_response(ℓs, ab_coeff, Dℓ_small)
        @test all(isfinite.(Δ_Dℓ_small))
        @test eltype(Δ_Dℓ_small) == Float64
        
        # Very large Dℓ values
        Dℓ_large = [1e10, 2e10, 1.5e10, 2.5e10, 2e10]
        Δ_Dℓ_large = CMBForegrounds.aberration_response(ℓs, ab_coeff, Dℓ_large)
        @test all(isfinite.(Δ_Dℓ_large))
        
        # Very small aberration coefficient
        ab_coeff_small = 1e-8
        Δ_Dℓ_small_ab = CMBForegrounds.aberration_response(ℓs, ab_coeff_small, Dℓ_small)
        @test all(isfinite.(Δ_Dℓ_small_ab))
        
        # High precision calculation
        ℓs_precise = [100π, 200π, 300π, 400π]
        Dℓ_precise = [π^2, ℯ^2, sqrt(2)^2, sqrt(3)^2]
        ab_coeff_precise = π/10000
        
        Δ_Dℓ_precise = CMBForegrounds.aberration_response(ℓs_precise, ab_coeff_precise, Dℓ_precise)
        @test all(isfinite.(Δ_Dℓ_precise))
    end
    
    @testset "Physical Realism - CMB Power Spectra" begin
        # Test with realistic CMB power spectrum
        ℓs = [50, 100, 200, 500, 1000, 2000, 3000]
        
        # Approximate CMB TT power spectrum shape
        Dℓ_cmb = @. 6000 * exp(-((ℓs - 200)/400)^2) / (ℓs/100 + 1) * (ℓs * (ℓs + 1) / (2π))
        
        # Typical aberration coefficients from observer motion
        ab_coeff_values = [1e-4, 1e-3, 1e-2]  # Range from small to large aberration
        
        for ab_coeff in ab_coeff_values
            Δ_Dℓ_cmb = CMBForegrounds.aberration_response(ℓs, ab_coeff, Dℓ_cmb)
            
            # Physical expectations
            @test all(isfinite.(Δ_Dℓ_cmb))
            @test length(Δ_Dℓ_cmb) == length(ℓs)
            
            # Aberration response should be finite and well-behaved
            @test all(isfinite.(Δ_Dℓ_cmb))
            @test all(abs.(Δ_Dℓ_cmb) .< 1e10)  # Not extreme
        end
    end
    
    @testset "Constant Dℓ Behavior" begin
        # Test with constant Dℓ (implies specific Cℓ ∝ 1/[ℓ(ℓ+1)])
        ℓs = [100, 200, 300, 400]
        Dℓ_const = [1000.0, 1000.0, 1000.0, 1000.0]
        ab_coeff = 1e-3
        
        Δ_Dℓ_const = CMBForegrounds.aberration_response(ℓs, ab_coeff, Dℓ_const)
        
        # For constant Dℓ, Cℓ ∝ 1/[ℓ(ℓ+1)], so dCℓ/dℓ < 0
        # Aberration = -ab_coeff * (negative) * ℓ²(ℓ+1)/(2π) = positive term
        @test all(isfinite.(Δ_Dℓ_const))
        @test all(Δ_Dℓ_const .!= 0.0)  # Should be non-zero
        
        # The response should still be proportional to ab_coeff
        Δ_Dℓ_double = CMBForegrounds.aberration_response(ℓs, 2*ab_coeff, Dℓ_const)
        @test all(Δ_Dℓ_double .≈ 2.0 .* Δ_Dℓ_const)
    end
    
    @testset "Power Law Dℓ Behavior" begin
        # Test with power law Dℓ ∝ ℓ^α
        ℓs = [100, 200, 400, 800]  # Powers of 2 for cleaner behavior
        α = -1.5
        A = 1000.0
        
        Dℓ_power = @. A * (ℓs / 100)^α
        ab_coeff = 1e-3
        
        Δ_Dℓ_power = CMBForegrounds.aberration_response(ℓs, ab_coeff, Dℓ_power)
        
        # Should be finite and well-behaved
        @test all(isfinite.(Δ_Dℓ_power))
        @test all(Δ_Dℓ_power .!= 0.0)  # Should be non-zero for power law
        
        # Should still scale with ab_coeff
        Δ_Dℓ_scaled = CMBForegrounds.aberration_response(ℓs, 3*ab_coeff, Dℓ_power)
        @test all(Δ_Dℓ_scaled .≈ 3.0 .* Δ_Dℓ_power)
    end
    
    @testset "Comparison with SSL Response" begin
        # Test that aberration response has similar structure to SSL but simpler
        ℓs = [100, 200, 300, 400]
        Dℓ = [1000.0, 800.0, 600.0, 400.0]  # Decreasing spectrum
        
        ab_coeff = 1e-3
        κ = 1e-3  # Same magnitude for comparison
        
        Δ_Dℓ_aberration = CMBForegrounds.aberration_response(ℓs, ab_coeff, Dℓ)
        Δ_Dℓ_ssl = CMBForegrounds.ssl_response(ℓs, κ, Dℓ)
        
        # Both should be finite
        @test all(isfinite.(Δ_Dℓ_aberration))
        @test all(isfinite.(Δ_Dℓ_ssl))
        
        # They should be different (SSL has additional amplitude term)
        @test Δ_Dℓ_aberration != Δ_Dℓ_ssl
        
        # But both should scale with their respective coefficients
        Δ_Dℓ_aberration_2x = CMBForegrounds.aberration_response(ℓs, 2*ab_coeff, Dℓ)
        @test all(Δ_Dℓ_aberration_2x .≈ 2.0 .* Δ_Dℓ_aberration)
    end
    
    @testset "Edge Cases and Robustness" begin
        # Test various edge cases
        
        # Very close ℓ values (challenging for derivatives)
        ℓs_close = [1000.0, 1000.1, 1000.2, 1000.3]
        Dℓ_close = [100.0, 100.05, 100.1, 100.15]
        ab_coeff = 1e-3
        
        Δ_Dℓ_close = CMBForegrounds.aberration_response(ℓs_close, ab_coeff, Dℓ_close)
        @test all(isfinite.(Δ_Dℓ_close))
        
        # Large ℓ values
        ℓs_large = [10000, 20000, 30000, 40000]
        Dℓ_large = [0.1, 0.08, 0.06, 0.04]
        
        Δ_Dℓ_large_l = CMBForegrounds.aberration_response(ℓs_large, ab_coeff, Dℓ_large)
        @test all(isfinite.(Δ_Dℓ_large_l))
        
        # Oscillatory Dℓ
        ℓs_osc = collect(100:50:300)
        Dℓ_osc = @. 1000 + 100 * sin(ℓs_osc / 50)
        
        Δ_Dℓ_osc = CMBForegrounds.aberration_response(ℓs_osc, ab_coeff, Dℓ_osc)
        @test all(isfinite.(Δ_Dℓ_osc))
        
        # Very large aberration coefficient
        ab_coeff_large = 0.1  # Unphysically large but mathematically valid
        ℓs_test = [100, 200, 300]
        Dℓ_test = [100.0, 200.0, 150.0]
        Δ_Dℓ_large_ab = CMBForegrounds.aberration_response(ℓs_test, ab_coeff_large, Dℓ_test)
        @test all(isfinite.(Δ_Dℓ_large_ab))
    end
    
    @testset "Sign Conventions" begin
        # Test sign conventions and physics
        ℓs = [100, 200, 300, 400]
        Dℓ = [1000.0, 800.0, 600.0, 400.0]  # Decreasing spectrum
        
        ab_coeff_pos = 1e-3
        ab_coeff_neg = -1e-3
        
        Δ_Dℓ_pos = CMBForegrounds.aberration_response(ℓs, ab_coeff_pos, Dℓ)
        Δ_Dℓ_neg = CMBForegrounds.aberration_response(ℓs, ab_coeff_neg, Dℓ)
        
        # Should be exactly opposite for opposite aberration coefficients
        @test all(Δ_Dℓ_pos .≈ -Δ_Dℓ_neg)
        
        # Check that the sign is consistent with the -ab_coeff factor
        dCℓ = CMBForegrounds.dCl_dell_from_Dl(ℓs, Dℓ)
        for i in 1:length(ℓs)
            ℓ = ℓs[i]
            pref = (ℓ * ℓ * (ℓ + 1)) / (2π)
            expected = -ab_coeff_pos * dCℓ[i] * pref
            @test Δ_Dℓ_pos[i] ≈ expected
        end
    end
    
    @testset "Multipole Scaling" begin
        # Test how the ℓ² factor affects different multipoles
        ℓs = [100, 200, 400, 800]  # Each double the previous
        
        # Use constant Cℓ so dCℓ/dℓ has predictable behavior
        # For constant Dℓ, Cℓ ∝ 1/[ℓ(ℓ+1)], so dCℓ/dℓ ∝ -1/ℓ² (approximately)
        Dℓ_const = [1000.0, 1000.0, 1000.0, 1000.0]
        ab_coeff = 1e-3
        
        Δ_Dℓ_mult = CMBForegrounds.aberration_response(ℓs, ab_coeff, Dℓ_const)
        
        # Aberration = -ab_coeff * dCℓ/dℓ * ℓ²(ℓ+1)/(2π)
        # For dCℓ/dℓ ∝ -1/ℓ², this becomes ∝ ℓ²(ℓ+1)/(ℓ²) = ℓ+1 ≈ ℓ for large ℓ
        # So aberration should increase roughly linearly with ℓ
        
        @test all(isfinite.(Δ_Dℓ_mult))
        @test all(Δ_Dℓ_mult .!= 0.0)
        
        # Should scale with multipole in some predictable way
        # (exact scaling depends on derivative behavior from finite differences)
        @test abs(Δ_Dℓ_mult[4]) > abs(Δ_Dℓ_mult[1])  # Higher ℓ should have larger response
    end
    
    @testset "Memory and Performance" begin
        # Test with larger arrays to check memory efficiency
        n = 1000
        ℓs_large = collect(10:10:10000)
        Dℓ_large = @. 1000 / (1 + (ℓs_large/500)^2)
        ab_coeff = 5e-4
        
        Δ_Dℓ_large = CMBForegrounds.aberration_response(ℓs_large, ab_coeff, Dℓ_large)
        
        # Basic correctness for large array
        @test length(Δ_Dℓ_large) == n
        @test all(isfinite.(Δ_Dℓ_large))
        @test eltype(Δ_Dℓ_large) == Float64
        
        # Should still scale with aberration coefficient
        Δ_Dℓ_scaled = CMBForegrounds.aberration_response(ℓs_large, 2*ab_coeff, Dℓ_large)
        @test all(Δ_Dℓ_scaled .≈ 2.0 .* Δ_Dℓ_large)
    end
    
    @testset "Consistency with Literature" begin
        # Test that the aberration formula matches expected form from literature
        ℓs = [200, 500, 1000]  # Representative multipoles
        
        # Use a realistic CMB-like spectrum at these scales
        Dℓ_realistic = [5000.0, 3000.0, 1000.0]  # Typical TT values in μK²
        ab_coeff = 1e-3  # Typical aberration coefficient
        
        Δ_Dℓ_lit = CMBForegrounds.aberration_response(ℓs, ab_coeff, Dℓ_realistic)
        
        # The magnitude should be ab_coeff times the derivative term times ℓ² factors
        dCℓ = CMBForegrounds.dCl_dell_from_Dl(ℓs, Dℓ_realistic)
        
        for i in 1:length(ℓs)
            ℓ = ℓs[i]
            
            # The formula term
            pref = (ℓ * ℓ * (ℓ + 1)) / (2π)
            expected_magnitude = abs(ab_coeff * dCℓ[i] * pref)
            
            @test abs(Δ_Dℓ_lit[i]) ≈ expected_magnitude
            @test sign(Δ_Dℓ_lit[i]) == -sign(ab_coeff * dCℓ[i] * pref)  # Check sign
        end
    end
    
    @testset "Numerical Derivatives Integration" begin
        # Test that the function properly handles the derivatives from dCl_dell_from_Dl
        ℓs = [100, 150, 200, 250, 300]  # Evenly spaced
        
        # Quadratic Dℓ for known derivative behavior
        a, b, c = 0.01, -2.0, 1000.0
        # Create Dℓ from a specific Cℓ form for predictable derivative
        Cℓ_quad = @. a * ℓs^2 + b * ℓs + c  
        Dℓ_quad = @. ℓs * (ℓs + 1) / (2π) * Cℓ_quad
        
        ab_coeff = 1e-3
        Δ_Dℓ_quad = CMBForegrounds.aberration_response(ℓs, ab_coeff, Dℓ_quad)
        
        # Should integrate properly with the derivative calculation
        @test all(isfinite.(Δ_Dℓ_quad))
        @test all(Δ_Dℓ_quad .!= 0.0)  # Should be non-zero
        
        # The middle points should be well-behaved (central differences work best there)
        @test all(isfinite.(Δ_Dℓ_quad[2:end-1]))
    end
    
    @testset "Derivative-Only Response" begin
        # Test that aberration only depends on derivatives (no amplitude term like SSL)
        ℓs = [100, 200, 300, 400]
        
        # Scale up the entire Dℓ spectrum by a factor
        Dℓ_base = [1000.0, 800.0, 600.0, 400.0]
        scale_factor = 2.0
        Dℓ_scaled = scale_factor .* Dℓ_base
        
        ab_coeff = 1e-3
        
        Δ_Dℓ_base = CMBForegrounds.aberration_response(ℓs, ab_coeff, Dℓ_base)
        Δ_Dℓ_scaled = CMBForegrounds.aberration_response(ℓs, ab_coeff, Dℓ_scaled)
        
        # Since Dℓ_scaled = scale_factor * Dℓ_base, we have:
        # dCℓ_scaled/dℓ = scale_factor * dCℓ_base/dℓ
        # So aberration response should also scale by scale_factor
        @test all(Δ_Dℓ_scaled .≈ scale_factor .* Δ_Dℓ_base)
        
        # This is different from SSL which has both derivative and amplitude terms
    end
end