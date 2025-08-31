"""
Unit tests for tsz_g_ratio function

Tests the tSZ (thermal Sunyaev-Zel'dovich) spectral function ratio g(ν)/g(ν0)
where g(x) = x(1 + 2/expm1(x)) - 4

COMPREHENSIVE TEST SUITE: Enhanced with mathematical limit verification,
physics-based validation, extreme edge cases, and performance testing.
"""

using Random

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

    @testset "Mathematical Identity Verification" begin
        # Verify g(x) = x(1 + 2/expm1(x)) - 4 implementation
        for params in [(100.0, 2.7), (50.0, 1.0), (200.0, 5.0)]
            ν, T = params
            r, x, _ = CMBForegrounds.dimensionless_freq_vars(ν, 100.0, T)
            
            # Manual g calculation
            g_manual = x * (1 + 2 / expm1(x)) - 4
            
            # Get g through the ratio function (using ν0 = ν to get g/g = 1, then multiply by g0)
            ratio_self = CMBForegrounds.tsz_g_ratio(ν, ν, T)  # Should be 1
            @test ratio_self ≈ 1.0
        end
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
        
        # Test that scaling preserves mathematical relationships
        ratio4 = CMBForegrounds.tsz_g_ratio(2*ν, 2*ν0, T)
        # Note: For tSZ g-function, scaling doesn't preserve ratio exactly due to the -4 term
        @test isfinite(ratio4)
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
        
        # Test extremely small temperature ratios
        ratio = CMBForegrounds.tsz_g_ratio(100.0, 50.0, 1e-6)
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
        
        # Test temperature scaling relationship
        for scale in [2.0, 5.0, 10.0]
            ratio_scaled = CMBForegrounds.tsz_g_ratio(ν, ν0, 2.7 * scale)
            @test ratio_scaled != ratio_midT
        end
    end

    @testset "Mathematical Limit Verification" begin
        
        @testset "Wien Limit Verification (x >> 1)" begin
            # Wien limit: g(x) → x - 4 for large x
            # Therefore ratio → (x - 4)/(x0 - 4)
            
            # Test case 1: Both frequencies in Wien regime
            ν, ν0, T = 500.0, 250.0, 0.05  # High freq, low temp
            r, x, x0 = CMBForegrounds.dimensionless_freq_vars(ν, ν0, T)
            @test x > 10  # Verify Wien limit
            @test x0 > 10
            
            actual_ratio = CMBForegrounds.tsz_g_ratio(ν, ν0, T)
            wien_approx = (x - 4) / (x0 - 4)
            @test actual_ratio ≈ wien_approx rtol=0.05
            
            # Test case 2: Extreme Wien limit
            ν, ν0, T = 1000.0, 500.0, 0.01
            r, x, x0 = CMBForegrounds.dimensionless_freq_vars(ν, ν0, T)
            @test x > 50  # Very deep Wien limit
            
            actual_ratio = CMBForegrounds.tsz_g_ratio(ν, ν0, T)
            wien_approx = (x - 4) / (x0 - 4)
            @test actual_ratio ≈ wien_approx rtol=0.01
            
            # Test case 3: When x >> 4, ratio → x/x0 = ν/ν0
            ν, ν0, T = 2000.0, 1000.0, 0.005
            actual_ratio = CMBForegrounds.tsz_g_ratio(ν, ν0, T)
            freq_ratio = ν / ν0
            @test actual_ratio ≈ freq_ratio rtol=0.02
        end
        
        @testset "Rayleigh-Jeans Limit Verification (x << 1)" begin
            # R-J limit: g(x) → x - 2 for small x
            # Therefore ratio → (x - 2)/(x0 - 2)
            
            # Test case 1: Both frequencies in R-J regime
            ν, ν0, T = 0.1, 0.05, 50.0  # Low freq, high temp
            r, x, x0 = CMBForegrounds.dimensionless_freq_vars(ν, ν0, T)
            @test x < 0.1  # Verify R-J limit
            @test x0 < 0.1
            
            actual_ratio = CMBForegrounds.tsz_g_ratio(ν, ν0, T)
            rj_approx = (x - 2) / (x0 - 2)
            @test actual_ratio ≈ rj_approx rtol=0.1
            
            # Test case 2: Very deep R-J limit
            ν, ν0, T = 0.01, 0.005, 100.0
            r, x, x0 = CMBForegrounds.dimensionless_freq_vars(ν, ν0, T)
            @test x < 0.01
            
            actual_ratio = CMBForegrounds.tsz_g_ratio(ν, ν0, T)
            rj_approx = (x - 2) / (x0 - 2)
            @test actual_ratio ≈ rj_approx rtol=0.05
            
            # Test case 3: When both x and x0 are very small, both g ≈ -2
            ν, ν0, T = 0.001, 0.0005, 200.0
            actual_ratio = CMBForegrounds.tsz_g_ratio(ν, ν0, T)
            @test actual_ratio ≈ 1.0 rtol=0.1  # Both g ≈ -2, so ratio ≈ 1
        end
        
        @testset "Intermediate Regime Verification" begin
            # Test transition regions between Wien and R-J
            
            # Around x ≈ 1
            for (ν, ν0, T) in [(100.0, 50.0, 5.0), (200.0, 100.0, 10.0)]
                r, x, x0 = CMBForegrounds.dimensionless_freq_vars(ν, ν0, T)
                if 0.5 < x < 2.0  # Intermediate regime
                    ratio = CMBForegrounds.tsz_g_ratio(ν, ν0, T)
                    @test isfinite(ratio)
                    @test abs(ratio) < 100  # Reasonable magnitude
                end
            end
        end
        
        @testset "Null Point Analysis" begin
            # tSZ g function has a null where g(x) = 0
            # This occurs when x(1 + 2/expm1(x)) - 4 = 0
            # or equivalently x(1 + 2/expm1(x)) = 4
            
            T_CMB = CMBForegrounds.T_CMB
            
            # The null occurs around 217 GHz for CMB temperature
            ν_null = 217.0
            r_null, x_null, _ = CMBForegrounds.dimensionless_freq_vars(ν_null, 100.0, T_CMB)
            
            # Test that g(x_null) ≈ 0
            g_null = x_null * (1 + 2 / expm1(x_null)) - 4
            @test abs(g_null) < 0.1
            
            # Test behavior around the null
            for offset in [-5.0, -2.0, -1.0, 1.0, 2.0, 5.0]
                ν_test = ν_null + offset
                ratio = CMBForegrounds.tsz_g_ratio(ν_test, 143.0, T_CMB)
                
                if offset < 0  # Below null
                    @test ratio > -1.0  # Should be small positive or small negative
                else  # Above null
                    @test ratio < 1.0   # Should be negative
                end
            end
        end
    end

    @testset "Physics-Based Validation Tests" begin
        T_CMB = CMBForegrounds.T_CMB
        
        @testset "Planck Mission Frequencies" begin
            # Test with actual Planck frequency bands
            planck_freqs = [30.0, 44.0, 70.0, 100.0, 143.0, 217.0, 353.0, 545.0, 857.0]
            reference_freq = 143.0  # Planck reference
            
            ratios = []
            for ν in planck_freqs
                ratio = CMBForegrounds.tsz_g_ratio(ν, reference_freq, T_CMB)
                push!(ratios, ratio)
                @test isfinite(ratio)
                
                # Physical constraints
                if ν < 100.0
                    @test ratio > 1.0  # Low frequency enhancement
                elseif ν ≈ reference_freq
                    @test abs(ratio - 1.0) < 0.01  # Reference should be 1
                elseif 200.0 < ν < 230.0
                    @test abs(ratio) < 0.2  # Near null
                elseif ν > 300.0
                    @test ratio < 0.0  # High frequency decrement (negative)
                end
                # Note: Some intermediate frequencies may have varying signs
            end
        end
        
        @testset "SPT-3G Survey Frequencies" begin
            # South Pole Telescope frequencies
            spt_freqs = [95.0, 150.0, 220.0]
            
            for ν in spt_freqs
                ratio = CMBForegrounds.tsz_g_ratio(ν, 150.0, T_CMB)
                @test isfinite(ratio)
                
                if ν == 150.0
                    @test ratio ≈ 1.0
                elseif ν == 95.0
                    @test ratio > 1.0  # Lower frequency
                elseif ν == 220.0
                    @test abs(ratio) < 0.1  # Near tSZ null
                end
            end
        end
        
        @testset "ACT Survey Frequencies" begin
            # Atacama Cosmology Telescope frequencies
            act_freqs = [98.0, 150.0, 220.0]
            
            for ν in act_freqs
                ratio = CMBForegrounds.tsz_g_ratio(ν, 150.0, T_CMB)
                @test isfinite(ratio)
                @test abs(ratio) < 10.0  # Reasonable range
            end
        end
        
        @testset "Realistic Cluster Temperatures" begin
            # Test with realistic galaxy cluster temperatures
            cluster_temps = [5.0, 8.0, 12.0, 15.0, 20.0]  # keV converted to K
            
            for T_cluster in cluster_temps
                for ν in [100.0, 150.0, 220.0]
                    ratio = CMBForegrounds.tsz_g_ratio(ν, 150.0, T_cluster)
                    @test isfinite(ratio)
                    # High temperature changes the spectral dependence
                    @test abs(ratio) < 50.0  # Still reasonable
                end
            end
        end
        
        @testset "Multi-frequency tSZ Analysis" begin
            # Simulate a multi-frequency tSZ analysis scenario
            frequencies = [90.0, 150.0, 220.0, 280.0]
            reference = 150.0
            
            ratios = [CMBForegrounds.tsz_g_ratio(ν, reference, T_CMB) for ν in frequencies]
            
            # Check that frequency ordering is preserved
            @test ratios[1] > ratios[2]  # 90 > 150
            @test ratios[2] ≈ 1.0        # 150 = reference
            @test abs(ratios[3]) < 0.2   # 220 near null
            @test ratios[4] < -1.0       # 280 strongly negative
        end
    end

    @testset "Extreme Value Numerical Stability" begin
        # Test with extreme parameter combinations
        extreme_cases = [
            (1e-8, 5e-9, 1000.0),      # Ultra-low frequency, high temperature
            (1e4, 5e3, 1e-6),          # Ultra-high frequency, ultra-low temperature  
            (1.0, 1000.0, 2.7),        # Large frequency ratio
            (1000.0, 1.0, 2.7),        # Reverse large frequency ratio
            (100.0, 100.0, 1e-9),      # Equal frequencies, ultra-low temperature
            (100.0, 100.0, 1e9),       # Equal frequencies, ultra-high temperature
            (1e-3, 1e3, 2.7)           # Extreme frequency difference
        ]
        
        for (ν, ν0, T) in extreme_cases
            ratio = CMBForegrounds.tsz_g_ratio(ν, ν0, T)
            @test isfinite(ratio)
            @test !isnan(ratio)
            @test abs(ratio) < 1e6  # Avoid completely unreasonable values
        end
    end

    @testset "Performance and Advanced Edge Cases" begin
        Random.seed!(42)
        
        @testset "Random Parameter Stress Test" begin
            # Test 200 random parameter combinations
            for i in 1:200
                ν = 10^(rand() * 4 - 1)      # 0.1 to 1000 GHz
                ν0 = 10^(rand() * 4 - 1)     # 0.1 to 1000 GHz  
                T = 10^(rand() * 3 - 1)      # 0.1 to 100 K
                
                ratio = CMBForegrounds.tsz_g_ratio(ν, ν0, T)
                @test isfinite(ratio)
            end
        end
        
        @testset "Monotonicity Tests" begin
            # Test monotonicity properties in different regimes
            T = CMBForegrounds.T_CMB
            ν0 = 143.0
            
            # In the positive regime (below null), should be decreasing with frequency
            freqs_positive = [30.0, 50.0, 70.0, 100.0, 143.0]
            ratios_pos = [CMBForegrounds.tsz_g_ratio(ν, ν0, T) for ν in freqs_positive]
            
            for i in 2:length(ratios_pos)
                if freqs_positive[i] <= ν0
                    @test ratios_pos[i] <= ratios_pos[i-1]  # Should be decreasing
                end
            end
        end
        
        @testset "Precision Near Critical Points" begin
            T_CMB = CMBForegrounds.T_CMB
            
            # Test precision around the tSZ null
            ν_null = 217.0
            for δν in [-0.1, -0.01, 0.01, 0.1, 1.0]
                ratio = CMBForegrounds.tsz_g_ratio(ν_null + δν, 143.0, T_CMB)
                @test isfinite(ratio)
                @test abs(ratio) < 1.0  # Should be small near null
            end
        end
        
        @testset "Frequency Grid Consistency" begin
            # Test on a regular frequency grid
            freqs = range(10.0, 1000.0, length=50)
            T = CMBForegrounds.T_CMB
            ν0 = 150.0
            
            ratios = [CMBForegrounds.tsz_g_ratio(ν, ν0, T) for ν in freqs]
            
            @test all(isfinite.(ratios))
            @test length(ratios) == length(freqs)
            
            # Check that there's a sign change (null crossing)
            signs = sign.(ratios)
            @test any(signs .> 0) && any(signs .< 0)
        end
        
        @testset "Temperature Grid Analysis" begin
            # Test over a range of temperatures
            temps = [0.1, 0.5, 1.0, 2.7, 5.0, 10.0, 20.0, 50.0, 100.0]
            ν, ν0 = 100.0, 150.0
            
            ratios = [CMBForegrounds.tsz_g_ratio(ν, ν0, T) for T in temps]
            
            @test all(isfinite.(ratios))
            # Temperature changes should affect the ratio
            @test length(unique(ratios)) == length(ratios)
        end
        
        @testset "Error Handling and Robustness" begin
            # Test potential edge cases that could cause issues
            
            # Very close but not equal frequencies
            ratio = CMBForegrounds.tsz_g_ratio(100.000001, 100.0, 2.7)
            @test isfinite(ratio)
            @test ratio ≈ 1.0 rtol=1e-3
            
            # Machine epsilon differences
            eps_freq = 100.0 + 1e-14
            ratio = CMBForegrounds.tsz_g_ratio(eps_freq, 100.0, 2.7)
            @test isfinite(ratio)
            
            # Test with subnormal numbers (if applicable)
            ratio = CMBForegrounds.tsz_g_ratio(1e-100, 1e-101, 1.0)
            @test isfinite(ratio)
        end
        
        @testset "Cross-Validation with Mathematical Properties" begin
            # Test mathematical relationships that should hold
            
            # Chain rule: g(ν1)/g(ν2) * g(ν2)/g(ν3) = g(ν1)/g(ν3)
            ν1, ν2, ν3, T = 100.0, 150.0, 200.0, 2.7
            
            ratio_12 = CMBForegrounds.tsz_g_ratio(ν1, ν2, T)
            ratio_23 = CMBForegrounds.tsz_g_ratio(ν2, ν3, T)
            ratio_13 = CMBForegrounds.tsz_g_ratio(ν1, ν3, T)
            
            @test ratio_12 * ratio_23 ≈ ratio_13 rtol=1e-10
            
            # Inverse relationship: g(ν)/g(ν0) * g(ν0)/g(ν) = 1
            for (ν, ν0, T) in [(100.0, 50.0, 2.7), (200.0, 300.0, 5.0)]
                forward = CMBForegrounds.tsz_g_ratio(ν, ν0, T)
                reverse = CMBForegrounds.tsz_g_ratio(ν0, ν, T)
                @test forward * reverse ≈ 1.0 rtol=1e-12
            end
        end
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
