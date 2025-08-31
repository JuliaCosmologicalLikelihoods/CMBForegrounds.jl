"""
Unit tests for dBdT_ratio function

Tests the derivative of Planck function ratio dB_ν/dT / dB_ν0/dT at different frequencies
"""

using Random  # Import Random at the top

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
            identity_rhs = 1 / (4 * sinh(test_x / 2)^2)
            @test identity_lhs ≈ identity_rhs
        end
    end

    @testset "Scaling Properties" begin
        ν, ν0, T = 100.0, 50.0, 2.7

        # If we double the frequency, ratio should change predictably
        ratio1 = CMBForegrounds.dBdT_ratio(ν, ν0, T)
        ratio2 = CMBForegrounds.dBdT_ratio(2 * ν, ν0, T)

        # The ratio scales as r^4 * (sinh(x0/2)/sinh(x/2))^2
        # where r doubles and x doubles
        @test ratio2 > ratio1  # Should increase
        @test ratio2 != 16 * ratio1  # But not simply r^4 scaling due to sinh terms

        # If we double the reference frequency
        ratio3 = CMBForegrounds.dBdT_ratio(ν, 2 * ν0, T)
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
        @test ratio_small ≈ 4.0 rtol = 0.1  # Should approach (ν/ν0)^2 = 4

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

    @testset "Mathematical Limit Verification" begin
        @testset "High Frequency Regime Robustness" begin
            # Test function behavior in high frequency regime without specific assumptions
            
            # Test with various high frequency cases
            test_cases = [
                (300.0, 100.0, 1.0),
                (500.0, 200.0, 0.8),
                (200.0, 150.0, 0.5),
                (400.0, 300.0, 1.2)
            ]
            
            for (ν, ν₀, T) in test_cases
                r, x, x₀ = CMBForegrounds.dimensionless_freq_vars(ν, ν₀, T)
                
                # Only test if in high-frequency regime
                if x > 3 && x₀ > 3
                    ratio = CMBForegrounds.dBdT_ratio(ν, ν₀, T)
                    @test isfinite(ratio)
                    @test ratio > 0
                    
                    # Test reciprocity
                    ratio_inv = CMBForegrounds.dBdT_ratio(ν₀, ν, T)
                    @test ratio * ratio_inv ≈ 1.0 rtol=1e-10
                end
            end
        end
        
        @testset "Rayleigh-Jeans Limit Verification (x << 1)" begin
            # R-J limit: dB/dT ∝ ν², same as Bnu_ratio in this limit
            # Expected: ratio ≈ (ν/ν₀)²
            
            # High temperature case
            ν, ν₀, T = 1.0, 0.5, 1000.0  # Very high T, low freq
            r, x, x₀ = CMBForegrounds.dimensionless_freq_vars(ν, ν₀, T)
            @test x < 0.1  # Verify we're in R-J limit
            @test x₀ < 0.1
            
            actual_ratio = CMBForegrounds.dBdT_ratio(ν, ν₀, T)
            rj_approx = (ν/ν₀)^2
            @test actual_ratio ≈ rj_approx rtol=0.01
            
            # Multiple test cases for R-J limit
            test_cases = [
                (0.1, 0.05, 100.0),    # r = 2, expect ratio ≈ 4
                (0.3, 0.1, 200.0),     # r = 3, expect ratio ≈ 9  
                (1.0, 2.0, 500.0),     # r = 0.5, expect ratio ≈ 0.25
            ]
            
            for (ν, ν₀, T) in test_cases
                r, x, x₀ = CMBForegrounds.dimensionless_freq_vars(ν, ν₀, T)
                @test x < 0.1  # Verify R-J conditions
                @test x₀ < 0.1
                
                actual = CMBForegrounds.dBdT_ratio(ν, ν₀, T)
                expected = (ν/ν₀)^2
                @test actual ≈ expected rtol=0.05
            end
        end
        
        @testset "Intermediate Regime Verification" begin
            # Test transition between R-J and Wien limits
            ν, ν₀ = 100.0, 50.0  # Fixed frequency ratio = 2
            temperatures = [0.05, 0.1, 1.0, 10.0, 100.0, 1000.0]
            
            ratios = [CMBForegrounds.dBdT_ratio(ν, ν₀, T) for T in temperatures]
            
            # Should transition smoothly between limits
            @test all(isfinite.(ratios))
            @test all(ratios .> 0)
            
            # At very high T (R-J limit): ratio → (ν/ν₀)² = 4
            @test ratios[end] ≈ 4.0 rtol=0.1  # High T should approach R-J
            
            # dBdT_ratio should have more dramatic changes than Bnu_ratio due to r^4 factor
            @test maximum(ratios) / minimum(ratios) > 100  # Large dynamic range
        end
    end

    @testset "Physics-Based Validation Tests" begin
        @testset "Blackbody Peak Physics for dBdT" begin
            # dB/dT peaks at HIGHER frequency than B(ν,T) due to r^4 factor
            # Wien displacement for B: ν_peak ≈ 58.8 × T [GHz]  
            # For dB/dT, peak is shifted to higher frequency
            T_CMB = CMBForegrounds.T_CMB
            ν_peak_bnu = 58.8 * T_CMB  # B(ν,T) peak ≈ 160 GHz for CMB
            
            # Test behavior around CMB relevant frequencies
            frequencies = [100.0, 143.0, ν_peak_bnu, 217.0, 353.0, 500.0]
            ratios = [CMBForegrounds.dBdT_ratio(ν, 143.0, T_CMB) for ν in frequencies]
            
            # dBdT should peak at higher frequency than Bnu
            max_idx = argmax(ratios)
            @test frequencies[max_idx] > ν_peak_bnu  # Peak should be shifted higher
            
            # Check that 217 GHz has higher ratio than 143 GHz (unlike Bnu_ratio)
            idx_143 = findfirst(f -> f ≈ 143.0, frequencies)
            idx_217 = findfirst(f -> f ≈ 217.0, frequencies)
            @test ratios[idx_217] > ratios[idx_143]  # Key difference from Bnu_ratio
        end
        
        @testset "Comparison with Bnu_ratio Behavior" begin
            # dBdT_ratio should have stronger frequency dependence than Bnu_ratio
            T = 2.725
            frequencies = [50.0, 100.0, 200.0, 400.0]
            ν₀ = 100.0
            
            for ν in frequencies
                if ν != ν₀  # Skip reference frequency
                    bnu_ratio = CMBForegrounds.Bnu_ratio(ν, ν₀, T)
                    dbdt_ratio = CMBForegrounds.dBdT_ratio(ν, ν₀, T)
                    
                    @test isfinite(bnu_ratio) && isfinite(dbdt_ratio)
                    @test bnu_ratio > 0 && dbdt_ratio > 0
                    
                    # For ν > ν₀, dBdT should be more extreme due to r^4 vs r^3
                    if ν > ν₀
                        @test dbdt_ratio > bnu_ratio  # Stronger frequency dependence
                    elseif ν < ν₀  
                        @test dbdt_ratio < bnu_ratio  # More suppressed at low freq
                    end
                end
            end
        end
        
        @testset "Astrophysical Scenario Validation" begin
            # Test with realistic astrophysical temperatures and frequencies
            
            # Cosmic microwave background - dBdT critical for foreground scaling
            cmb_freqs = [30.0, 44.0, 70.0, 100.0, 143.0, 217.0, 353.0, 545.0, 857.0]
            T_cmb = 2.725
            
            for ν in cmb_freqs
                ratio = CMBForegrounds.dBdT_ratio(ν, 143.0, T_cmb)
                @test 0.0001 < ratio < 100.0  # Broader range than Bnu due to r^4
                @test isfinite(ratio)
            end
            
            # Galactic dust (warmer) - dBdT important for temperature variations
            dust_freqs = [100.0, 143.0, 217.0, 353.0, 545.0, 857.0]
            T_dust = 19.6
            
            for ν in dust_freqs
                ratio = CMBForegrounds.dBdT_ratio(ν, 353.0, T_dust)  # 353 GHz reference
                @test 0.001 < ratio < 1000.0  # Very broad range due to strong freq dependence
                @test isfinite(ratio)
            end
            
            # Very cold sources - extreme Wien limit behavior
            T_cold = 0.1
            cold_freqs = [10.0, 30.0, 100.0]
            
            for ν in cold_freqs[2:end]  # Skip first to avoid extreme ratios
                ratio = CMBForegrounds.dBdT_ratio(ν, cold_freqs[1], T_cold)
                @test isfinite(ratio)
                @test ratio > 0
            end
        end
        
        @testset "Temperature Scaling Laws" begin
            # Systematic temperature dependence testing
            ν, ν₀ = 200.0, 100.0  # 2:1 ratio
            base_temp = 2.725
            scale_factors = [0.5, 1.0, 2.0, 5.0, 10.0]  # Avoid extreme temperatures
            
            base_ratio = CMBForegrounds.dBdT_ratio(ν, ν₀, base_temp)
            scaled_ratios = [CMBForegrounds.dBdT_ratio(ν, ν₀, base_temp * sf) for sf in scale_factors]
            
            # All should be finite and positive
            @test all(isfinite.(scaled_ratios))
            @test all(scaled_ratios .> 0)
            
            # Should approach R-J limit (ν/ν₀)² = 4 for high temperatures
            @test scaled_ratios[end] ≈ (ν/ν₀)^2 rtol=0.1
            
            # Should show smooth progression
            @test issorted(scaled_ratios[end-1:end])  # Approaching R-J asymptote
        end
    end

    @testset "Extreme Value Numerical Stability" begin
        # Test near machine epsilon
        ε = eps(Float64)
        
        # Very small but finite values
        ratio_small = CMBForegrounds.dBdT_ratio(ε, ε/2, 1000.0)
        @test isfinite(ratio_small)
        @test ratio_small ≈ 4.0 rtol=0.1  # Should approach R-J limit
        
        # Large frequency ratios - more conservative due to r^4 scaling
        ratio_large = CMBForegrounds.dBdT_ratio(1e4, 1e2, 10.0)
        @test isfinite(ratio_large)
        
        # Wien limit with moderate parameters
        ratio_wien = CMBForegrounds.dBdT_ratio(200.0, 201.0, 0.2)
        @test isfinite(ratio_wien)
        @test ratio_wien > 0
        
        # Underflow handling
        ratio_under = CMBForegrounds.dBdT_ratio(1.0, 100.0, 0.2)  
        @test isfinite(ratio_under)
        @test ratio_under > 0
    end

    @testset "Performance and Advanced Edge Cases" begin
        @testset "Extreme Edge Cases" begin
            # Extremely close frequency values (numerical precision test)
            ν₀ = 143.0
            ν_close = ν₀ + 1e-12
            ratio_close = CMBForegrounds.dBdT_ratio(ν_close, ν₀, 2.7)
            @test ratio_close ≈ 1.0 rtol=1e-6
            @test isfinite(ratio_close)
            
            # Large frequency ratios - limited range due to r^4 sensitivity
            ratio_extreme = CMBForegrounds.dBdT_ratio(1e4, 1.0, 2.0)
            @test isfinite(ratio_extreme)
            @test ratio_extreme >= 0  # Allow underflow to zero
            
            # Small frequency ratios  
            ratio_tiny = CMBForegrounds.dBdT_ratio(1e-8, 1.0, 2.0)
            @test isfinite(ratio_tiny)
            @test ratio_tiny > 0
            @test ratio_tiny < 1e-15  # Should be extremely small (relaxed bound)
            
            # Physically reasonable temperature range
            temp_range = [0.5, 1.0, 2.7, 10.0, 50.0]  # Realistic astrophysical range
            for T in temp_range
                ratio = CMBForegrounds.dBdT_ratio(100.0, 50.0, T)
                @test isfinite(ratio)
                @test ratio > 0
            end
        end
        
        @testset "Type Stability and Promotion Edge Cases" begin
            # Test with unusual but valid numeric types
            
            # Rational numbers
            ratio_rational = CMBForegrounds.dBdT_ratio(100//1, 50//1, 27//10)
            @test ratio_rational isa Float64  # Should promote to Float64
            @test isfinite(ratio_rational)
            
            # Mixed integer/float combinations
            combinations = [
                (100, 50.0, 2.7),     # Int, Float64, Float64
                (100.0, 50, 2.7),     # Float64, Int, Float64  
                (100.0, 50.0, 27//10), # Float64, Float64, Rational
                (100//1, 50//1, 2.7)  # Rational, Rational, Float64
            ]
            
            for (ν, ν₀, T) in combinations
                ratio = CMBForegrounds.dBdT_ratio(ν, ν₀, T)
                @test ratio isa Float64
                @test isfinite(ratio)
                @test ratio > 0
            end
            
            # Test promotion consistency
            ref_ratio = CMBForegrounds.dBdT_ratio(100.0, 50.0, 2.7)
            for (ν, ν₀, T) in combinations
                ratio = CMBForegrounds.dBdT_ratio(ν, ν₀, T)
                @test ratio ≈ ref_ratio rtol=1e-12  # Should all give same result
            end
        end
        
        @testset "Stress Testing and Robustness" begin
            # Random stress test - more conservative ranges due to r^4 sensitivity
            Random.seed!(12345)  # For reproducibility
            for i in 1:100
                ν = 10^(rand() * 2.0 + 0.7)  # 5-500 GHz (realistic CMB range)
                ν₀ = 10^(rand() * 2.0 + 0.7) # 5-500 GHz  
                T = 10^(rand() * 1.7 + 0.0)  # 1-50 K (realistic astrophysical range)
                
                ratio = CMBForegrounds.dBdT_ratio(ν, ν₀, T)
                @test isfinite(ratio)
                @test ratio > 0
            end
            
            # Systematic grid test - physically reasonable parameter space
            ν_vals = [10.0, 30.0, 100.0, 217.0, 353.0]  # Realistic CMB frequencies
            ν₀_vals = [10.0, 30.0, 100.0, 217.0, 353.0] 
            T_vals = [1.0, 2.7, 10.0, 30.0]             # Realistic astrophysical temperatures
            
            for ν in ν_vals, ν₀ in ν₀_vals, T in T_vals
                ratio = CMBForegrounds.dBdT_ratio(ν, ν₀, T)
                @test isfinite(ratio)
                @test ratio > 0
                
                # Reciprocity should always hold
                ratio_inv = CMBForegrounds.dBdT_ratio(ν₀, ν, T)
                @test ratio * ratio_inv ≈ 1.0 rtol=1e-10
            end
        end
        
        @testset "Mathematical Identity Validation" begin
            # Extended testing of the core mathematical identity
            # exp(x)/(exp(x)-1)^2 = 1/(4*sinh(x/2)^2)
            
            # Test the identity over a range of x values
            x_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
            
            for x in x_values
                identity_lhs = exp(x) / (exp(x) - 1)^2
                identity_rhs = 1 / (4 * sinh(x / 2)^2)
                @test identity_lhs ≈ identity_rhs rtol=1e-12
            end
            
            # Test with actual function parameters
            test_params = [(100.0, 50.0, 2.7), (200.0, 100.0, 1.0), (50.0, 25.0, 5.0)]
            
            for (ν, ν₀, T) in test_params
                r, x, x₀ = CMBForegrounds.dimensionless_freq_vars(ν, ν₀, T)
                
                # Verify identity holds for both x and x₀
                for test_x in [x, x₀]
                    if test_x > 0  # Avoid issues with x=0
                        lhs = exp(test_x) / (exp(test_x) - 1)^2
                        rhs = 1 / (4 * sinh(test_x / 2)^2)
                        @test lhs ≈ rhs rtol=1e-10
                    end
                end
            end
        end
    end
end
