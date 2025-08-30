"""
Unit tests for dBdT_ratio function

Tests the derivative of Planck function ratio dB_ν/dT / dB_ν0/dT at different frequencies
"""

using Test
using CMBForegrounds

@testset "dBdT_ratio() Unit Tests" begin
    
    @testset "Basic Functionality" begin
        # Test with simple values
        ν, ν0, T = 100.0, 50.0, 2.7
        ratio = CMBForegrounds.dBdT_ratio(ν, ν0, T)
        
        # Basic output tests
        @test ratio isa Number
        @test isfinite(ratio)
        @test ratio > 0  # dB/dT ratios should always be positive
    end
    
    @testset "Mathematical Properties" begin
        # Test with ν = ν0 (should give ratio = 1)
        ratio = CMBForegrounds.dBdT_ratio(143.0, 143.0, 2.7)
        @test ratio ≈ 1.0
        
        # Test behavior at CMB temperatures - dBdT has different frequency dependence than Bnu
        T = 2.725
        ratio_low = CMBForegrounds.dBdT_ratio(100.0, 143.0, T)
        ratio_ref = CMBForegrounds.dBdT_ratio(143.0, 143.0, T) 
        ratio_high = CMBForegrounds.dBdT_ratio(217.0, 143.0, T)
        
        @test ratio_ref ≈ 1.0
        # dBdT_ratio has different behavior - it can increase then decrease
        @test ratio_low < ratio_ref    # 100 GHz lower than reference
        @test ratio_high > ratio_ref   # 217 GHz higher than reference (unlike Bnu_ratio)
    end
    
    @testset "Consistency with Formula" begin
        # Test that the function matches the mathematical definition
        ν, ν0, T = 100.0, 50.0, 2.7
        
        # Get dimensionless variables
        r, x, x0 = CMBForegrounds.dimensionless_freq_vars(ν, ν0, T)
        
        # Manual calculation: r^4 * (sinh(x0/2))^2 / (sinh(x/2))^2
        s0 = sinh(x0 / 2)
        s = sinh(x / 2)
        expected_ratio = r^4 * (s0 * s0) / (s * s)
        actual_ratio = CMBForegrounds.dBdT_ratio(ν, ν0, T)
        
        @test actual_ratio ≈ expected_ratio
    end
    
    @testset "Mathematical Identity Verification" begin
        # Verify the mathematical identity exp(x)/(exp(x)-1)^2 = 1/(4*sinh(x/2)^2)
        # This identity is used in the derivation but not directly in the computation
        ν, ν0, T = 100.0, 50.0, 2.7
        r, x, x0 = CMBForegrounds.dimensionless_freq_vars(ν, ν0, T)
        
        # Verify the mathematical identity for both x and x0
        for test_x in [x, x0]
            identity_lhs = exp(test_x) / (exp(test_x) - 1)^2
            identity_rhs = 1 / (4 * sinh(test_x/2)^2)
            @test identity_lhs ≈ identity_rhs
        end
    end
    
    @testset "Scaling Properties" begin
        ν, ν0, T = 100.0, 50.0, 2.7
        
        # If we double the frequency, ratio should change predictably
        ratio1 = CMBForegrounds.dBdT_ratio(ν, ν0, T)
        ratio2 = CMBForegrounds.dBdT_ratio(2*ν, ν0, T)
        
        # The ratio scales as r^4 * (sinh(x0/2)/sinh(x/2))^2
        # where r doubles and x doubles
        @test ratio2 > ratio1  # Should increase
        @test ratio2 != 16 * ratio1  # But not simply r^4 scaling due to sinh terms
        
        # If we double the reference frequency
        ratio3 = CMBForegrounds.dBdT_ratio(ν, 2*ν0, T)
        @test ratio3 < ratio1  # Should decrease (lower reference frequency)
    end
    
    @testset "Type Stability" begin
        # Test with Float64
        ratio = CMBForegrounds.dBdT_ratio(100.0, 50.0, 2.7)
        @test ratio isa Float64
        
        # Test with Int (should promote to Float64)
        ratio = CMBForegrounds.dBdT_ratio(100, 50, 3)
        @test ratio isa Float64
        
        # Test with mixed types
        ratio = CMBForegrounds.dBdT_ratio(100, 50.0, 2.7)
        @test ratio isa Float64
    end
    
    @testset "Edge Cases" begin
        # Test with very small frequencies (Rayleigh-Jeans limit)
        # In R-J limit: dB/dT ∝ ν^2, so ratio → (ν/ν0)^2
        ratio_small = CMBForegrounds.dBdT_ratio(0.001, 0.0005, 2.7)
        @test ratio_small ≈ 4.0 rtol=0.1  # Should approach (ν/ν0)^2 = 4
        
        # Test with large frequencies (Wien limit)
        ratio_large = CMBForegrounds.dBdT_ratio(1000.0, 500.0, 0.1)
        @test isfinite(ratio_large)
        @test ratio_large > 0
        
        # Test with very small temperature (Wien regime)
        ratio_wien = CMBForegrounds.dBdT_ratio(100.0, 50.0, 0.01)
        @test isfinite(ratio_wien)
        @test ratio_wien > 0
    end
    
    @testset "Physical Consistency" begin
        # Test with realistic CMB values
        T_CMB = CMBForegrounds.T_CMB
        
        # Planck frequencies
        ratio_100 = CMBForegrounds.dBdT_ratio(100.0, 143.0, T_CMB)
        ratio_143 = CMBForegrounds.dBdT_ratio(143.0, 143.0, T_CMB)
        ratio_217 = CMBForegrounds.dBdT_ratio(217.0, 143.0, T_CMB)
        ratio_353 = CMBForegrounds.dBdT_ratio(353.0, 143.0, T_CMB)
        
        # Physical expectations - dBdT has different behavior than Bnu
        @test ratio_143 ≈ 1.0
        @test ratio_100 < ratio_143      # Below reference
        @test ratio_217 > ratio_143      # Above reference (different from Bnu_ratio)
        @test ratio_353 < ratio_217      # Very high freq starts to decrease
        
        # Check reasonable values for CMB physics
        @test 0.3 < ratio_100 < 1.0      # Below reference
        @test 1.0 < ratio_217 < 2.0      # Above reference but reasonable
        @test 0.5 < ratio_353 < 1.0      # High freq, can be lower again
    end
    
    @testset "Numerical Stability" begin
        # Test near x = 0 (should handle sinh correctly)
        ratio = CMBForegrounds.dBdT_ratio(1e-6, 5e-7, 1000.0)  # Very low freq, high T
        @test isfinite(ratio)
        @test ratio > 0
        
        # Test for large x values (should not overflow)
        ratio = CMBForegrounds.dBdT_ratio(1000.0, 100.0, 0.1)
        @test isfinite(ratio)
        @test ratio > 0
        
        # Test that sinh(x/2) doesn't underflow for small x
        ratio = CMBForegrounds.dBdT_ratio(1e-8, 5e-9, 100.0)
        @test isfinite(ratio)
        @test ratio > 0
    end
    
    @testset "Symmetry Properties" begin
        # Test reciprocal relationship: dBdT_ratio(ν0, ν, T) = 1/dBdT_ratio(ν, ν0, T)
        ν, ν0, T = 100.0, 50.0, 2.7
        ratio_forward = CMBForegrounds.dBdT_ratio(ν, ν0, T)
        ratio_reverse = CMBForegrounds.dBdT_ratio(ν0, ν, T)
        
        @test ratio_forward * ratio_reverse ≈ 1.0
    end
    
    @testset "Relationship to Bnu_ratio" begin
        # dB/dT and B_ν are related but different - test they give different results
        ν, ν0, T = 100.0, 50.0, 2.7
        
        bnu_ratio = CMBForegrounds.Bnu_ratio(ν, ν0, T)
        dbdt_ratio = CMBForegrounds.dBdT_ratio(ν, ν0, T)
        
        # They should be different (unless at very specific conditions)
        @test bnu_ratio != dbdt_ratio
        
        # But both should be positive and finite
        @test bnu_ratio > 0 && isfinite(bnu_ratio)
        @test dbdt_ratio > 0 && isfinite(dbdt_ratio)
        
        # dBdT typically has stronger frequency dependence (r^4 vs r^3)
        # So for ν > ν0, dBdT_ratio should be larger than Bnu_ratio
        if ν > ν0
            @test dbdt_ratio > bnu_ratio
        end
    end
    
    @testset "Temperature Scaling" begin
        # Test how the ratio changes with temperature
        ν, ν0 = 100.0, 50.0
        
        ratio_lowT = CMBForegrounds.dBdT_ratio(ν, ν0, 1.0)
        ratio_midT = CMBForegrounds.dBdT_ratio(ν, ν0, 2.7)
        ratio_highT = CMBForegrounds.dBdT_ratio(ν, ν0, 10.0)
        
        # All should be finite and positive
        @test all(isfinite.([ratio_lowT, ratio_midT, ratio_highT]))
        @test all([ratio_lowT, ratio_midT, ratio_highT] .> 0)
        
        # The exact ordering depends on frequency regime, but all should be reasonable
        @test 0.1 < ratio_lowT < 1000.0
        @test 0.1 < ratio_midT < 1000.0  
        @test 0.1 < ratio_highT < 1000.0
    end
end