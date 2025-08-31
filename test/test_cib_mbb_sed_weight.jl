"""
Unit tests for cib_mbb_sed_weight function

Tests the CIB (Cosmic Infrared Background) modified blackbody spectral energy distribution weight
Weight = r^β * B_ν(T_dust)/B_ν0(T_dust) / (dB/dT)_ν/(dB/dT)_ν0 (T_CMB)

COMPREHENSIVE TEST SUITE: Enhanced with mathematical limit verification,
component decomposition testing, physics-based validation, and performance testing.
"""

using Random

@testset "cib_mbb_sed_weight() Unit Tests" begin

    @testset "Basic Functionality" begin
        # Test with typical CIB parameters
        β, T_dust, ν0, ν = 1.5, 25.0, 143.0, 217.0
        weight = CMBForegrounds.cib_mbb_sed_weight(β, T_dust, ν0, ν)

        # Basic output tests
        @test weight isa Number
        @test isfinite(weight)
        @test weight > 0  # CIB weights should always be positive
    end

    @testset "Mathematical Properties" begin
        # Test with ν = ν0 (should give weight = 1)
        β, T_dust = 1.5, 25.0
        weight = CMBForegrounds.cib_mbb_sed_weight(β, T_dust, 143.0, 143.0)
        @test weight ≈ 1.0

        # Test frequency dependence - higher frequency should give higher weight
        T_dust = 25.0  # Warm dust temperature
        weight_100 = CMBForegrounds.cib_mbb_sed_weight(1.5, T_dust, 143.0, 100.0)
        weight_143 = CMBForegrounds.cib_mbb_sed_weight(1.5, T_dust, 143.0, 143.0)
        weight_217 = CMBForegrounds.cib_mbb_sed_weight(1.5, T_dust, 143.0, 217.0)
        weight_353 = CMBForegrounds.cib_mbb_sed_weight(1.5, T_dust, 143.0, 353.0)

        @test weight_143 ≈ 1.0
        @test weight_100 < weight_143 < weight_217 < weight_353  # Should increase with frequency
    end

    @testset "Consistency with Formula" begin
        # Test that the function matches the mathematical definition
        β, T_dust, ν0, ν = 1.5, 25.0, 143.0, 217.0
        T_CMB = CMBForegrounds.T_CMB

        # Manual calculation: r^β * Bnu_ratio(dust) / dBdT_ratio(CMB)
        r = ν / ν0
        bnu_ratio = CMBForegrounds.Bnu_ratio(ν, ν0, T_dust)
        dbdt_ratio = CMBForegrounds.dBdT_ratio(ν, ν0, T_CMB)
        expected_weight = r^β * bnu_ratio / dbdt_ratio
        actual_weight = CMBForegrounds.cib_mbb_sed_weight(β, T_dust, ν0, ν)

        @test actual_weight ≈ expected_weight
    end

    @testset "Default T_CMB Parameter" begin
        # Test that default T_CMB works correctly
        β, T_dust, ν0, ν = 1.5, 25.0, 143.0, 217.0

        # With explicit T_CMB
        weight_explicit = CMBForegrounds.cib_mbb_sed_weight(β, T_dust, ν0, ν, T_CMB=CMBForegrounds.T_CMB)
        # With default T_CMB
        weight_default = CMBForegrounds.cib_mbb_sed_weight(β, T_dust, ν0, ν)

        @test weight_explicit ≈ weight_default

        # Test with different T_CMB
        weight_different = CMBForegrounds.cib_mbb_sed_weight(β, T_dust, ν0, ν, T_CMB=3.0)
        @test weight_different != weight_default
    end

    @testset "Beta Parameter Effects" begin
        # Test how different β values affect the spectral index
        T_dust, ν0, ν = 25.0, 143.0, 217.0

        weight_β1 = CMBForegrounds.cib_mbb_sed_weight(1.0, T_dust, ν0, ν)
        weight_β15 = CMBForegrounds.cib_mbb_sed_weight(1.5, T_dust, ν0, ν)
        weight_β2 = CMBForegrounds.cib_mbb_sed_weight(2.0, T_dust, ν0, ν)

        # Higher β should give more weight at higher frequencies
        @test weight_β1 < weight_β15 < weight_β2

        # Test β = 0 (no frequency dependence from power law)
        weight_β0_100 = CMBForegrounds.cib_mbb_sed_weight(0.0, T_dust, 143.0, 100.0)
        weight_β0_217 = CMBForegrounds.cib_mbb_sed_weight(0.0, T_dust, 143.0, 217.0)

        # r^0 = 1, so only blackbody effects remain
        r_100 = 100.0 / 143.0
        r_217 = 217.0 / 143.0
        @test weight_β0_217 / weight_β0_100 ≈ (CMBForegrounds.Bnu_ratio(217.0, 100.0, T_dust) /
                                               CMBForegrounds.dBdT_ratio(217.0, 100.0, CMBForegrounds.T_CMB)) rtol = 1e-10
    end

    @testset "Dust Temperature Effects" begin
        # Test how dust temperature affects the spectral shape
        β, ν0, ν = 1.5, 143.0, 353.0  # High frequency where dust dominates

        weight_cold = CMBForegrounds.cib_mbb_sed_weight(β, 15.0, ν0, ν)  # Cold dust
        weight_warm = CMBForegrounds.cib_mbb_sed_weight(β, 25.0, ν0, ν)  # Warm dust
        weight_hot = CMBForegrounds.cib_mbb_sed_weight(β, 35.0, ν0, ν)   # Hot dust

        # All should be positive and finite
        @test all([weight_cold, weight_warm, weight_hot] .> 0)
        @test all(isfinite.([weight_cold, weight_warm, weight_hot]))

        # At high frequencies, warmer dust should generally give higher weights
        # (though this depends on the balance of Bnu_ratio and the reference frequency)
        @test weight_cold != weight_warm != weight_hot
    end

    @testset "Type Stability" begin
        # Test with Float64
        weight = CMBForegrounds.cib_mbb_sed_weight(1.5, 25.0, 143.0, 217.0)
        @test weight isa Float64

        # Test with Int (should promote to Float64)
        weight = CMBForegrounds.cib_mbb_sed_weight(1, 25, 143, 217)
        @test weight isa Float64

        # Test with mixed types
        weight = CMBForegrounds.cib_mbb_sed_weight(1.5, 25, 143.0, 217)
        @test weight isa Float64
    end

    @testset "Edge Cases" begin
        # Test with very small frequencies
        weight_small = CMBForegrounds.cib_mbb_sed_weight(1.5, 25.0, 1.0, 0.5)
        @test isfinite(weight_small)
        @test weight_small > 0

        # Test with large frequencies
        weight_large = CMBForegrounds.cib_mbb_sed_weight(1.5, 25.0, 100.0, 1000.0)
        @test isfinite(weight_large)
        @test weight_large > 0

        # Test with extreme β values
        weight_β_small = CMBForegrounds.cib_mbb_sed_weight(0.1, 25.0, 143.0, 217.0)
        weight_β_large = CMBForegrounds.cib_mbb_sed_weight(3.0, 25.0, 143.0, 217.0)
        @test isfinite(weight_β_small) && weight_β_small > 0
        @test isfinite(weight_β_large) && weight_β_large > 0

        # Test with extreme dust temperatures
        weight_very_cold = CMBForegrounds.cib_mbb_sed_weight(1.5, 5.0, 143.0, 217.0)
        weight_very_hot = CMBForegrounds.cib_mbb_sed_weight(1.5, 100.0, 143.0, 217.0)
        @test isfinite(weight_very_cold) && weight_very_cold > 0
        @test isfinite(weight_very_hot) && weight_very_hot > 0
    end

    @testset "Physical Consistency" begin
        # Test with realistic CIB parameters
        β_typical = 1.6      # Typical CIB emissivity index
        T_dust = 24.0        # Typical CIB dust temperature

        # Planck frequencies
        weight_100 = CMBForegrounds.cib_mbb_sed_weight(β_typical, T_dust, 143.0, 100.0)
        weight_143 = CMBForegrounds.cib_mbb_sed_weight(β_typical, T_dust, 143.0, 143.0)
        weight_217 = CMBForegrounds.cib_mbb_sed_weight(β_typical, T_dust, 143.0, 217.0)
        weight_353 = CMBForegrounds.cib_mbb_sed_weight(β_typical, T_dust, 143.0, 353.0)
        weight_545 = CMBForegrounds.cib_mbb_sed_weight(β_typical, T_dust, 143.0, 545.0)

        # Physical expectations for CIB
        @test weight_143 ≈ 1.0
        @test weight_100 < weight_143 < weight_217 < weight_353 < weight_545

        # Check reasonable ranges for CIB physics
        @test 0.1 < weight_100 < 1.0      # Below reference frequency
        @test 1.0 < weight_217 < 10.0     # Moderate increase
        @test 10.0 < weight_353 < 50.0    # Strong increase at high freq
        @test 100.0 < weight_545 < 1000.0 # Very strong at far-IR
    end

    @testset "Scaling Properties" begin
        # Test how the weight scales with different parameters
        β, T_dust, ν0 = 1.5, 25.0, 143.0

        # If we double the frequency, weight should increase
        weight1 = CMBForegrounds.cib_mbb_sed_weight(β, T_dust, ν0, 200.0)
        weight2 = CMBForegrounds.cib_mbb_sed_weight(β, T_dust, ν0, 400.0)
        @test weight2 > weight1

        # The scaling includes r^β, so doubling frequency gives factor 2^β from power law
        r_factor = (400.0 / 200.0)^β  # = 2^1.5 ≈ 2.83
        # But total scaling also includes blackbody effects, so not exactly r^β
        @test weight2 / weight1 != r_factor  # Blackbody effects matter
        @test weight2 / weight1 > 1.0  # But should still increase
    end

    @testset "Reference Frequency Effects" begin
        # Test how changing reference frequency affects weights
        β, T_dust, ν = 1.5, 25.0, 353.0

        weight_ref100 = CMBForegrounds.cib_mbb_sed_weight(β, T_dust, 100.0, ν)
        weight_ref143 = CMBForegrounds.cib_mbb_sed_weight(β, T_dust, 143.0, ν)
        weight_ref217 = CMBForegrounds.cib_mbb_sed_weight(β, T_dust, 217.0, ν)

        # All should be positive and finite
        @test all([weight_ref100, weight_ref143, weight_ref217] .> 0)
        @test all(isfinite.([weight_ref100, weight_ref143, weight_ref217]))

        # Different reference frequencies should give different weights
        @test weight_ref100 != weight_ref143 != weight_ref217
    end

    @testset "Numerical Stability" begin
        # Test numerical stability for extreme parameter combinations

        # Very small β
        weight = CMBForegrounds.cib_mbb_sed_weight(1e-3, 25.0, 143.0, 217.0)
        @test isfinite(weight) && weight > 0

        # Very large β
        weight = CMBForegrounds.cib_mbb_sed_weight(5.0, 25.0, 143.0, 217.0)
        @test isfinite(weight) && weight > 0

        # Very close frequencies (r ≈ 1)
        weight = CMBForegrounds.cib_mbb_sed_weight(1.5, 25.0, 143.0, 143.001)
        @test isfinite(weight) && weight > 0
        @test weight ≈ 1.0 rtol = 0.01  # Should be close to 1

        # Very different frequencies
        weight = CMBForegrounds.cib_mbb_sed_weight(1.5, 25.0, 30.0, 857.0)
        @test isfinite(weight) && weight > 0
    end

    @testset "Promote Function Effects" begin
        # Test that promote() works correctly with mixed input types

        # All integers
        weight_int = CMBForegrounds.cib_mbb_sed_weight(2, 25, 143, 217)
        # All floats
        weight_float = CMBForegrounds.cib_mbb_sed_weight(2.0, 25.0, 143.0, 217.0)

        @test weight_int ≈ weight_float
        @test typeof(weight_int) == typeof(weight_float) == Float64

        # Mixed types
        weight_mixed = CMBForegrounds.cib_mbb_sed_weight(2, 25.0, 143, 217.0, T_CMB=3)
        @test weight_mixed isa Float64
        @test isfinite(weight_mixed) && weight_mixed > 0
    end

    @testset "Component Decomposition Testing" begin
        # Test individual components: r^β, Bnu_ratio(dust), dBdT_ratio(CMB)
        β, T_dust, ν0, ν = 1.5, 25.0, 143.0, 217.0
        T_CMB = CMBForegrounds.T_CMB
        
        # Manual component calculation
        r = ν / ν0
        power_law_component = r^β
        bnu_component = CMBForegrounds.Bnu_ratio(ν, ν0, T_dust)
        dbdt_component = CMBForegrounds.dBdT_ratio(ν, ν0, T_CMB)
        expected_weight = power_law_component * bnu_component / dbdt_component
        
        actual_weight = CMBForegrounds.cib_mbb_sed_weight(β, T_dust, ν0, ν)
        @test actual_weight ≈ expected_weight
        
        # Test individual component ranges
        @test power_law_component > 1.0  # ν > ν0, β > 0
        @test bnu_component > 1.0        # Higher freq at warm dust temp
        @test dbdt_component > 1.0       # Higher freq at CMB temp
        @test expected_weight > 0.0      # All positive components
    end

    @testset "Mathematical Identity Verification" begin
        # Verify the formula implementation with multiple parameter sets
        test_cases = [
            (1.0, 20.0, 100.0, 200.0),
            (2.0, 30.0, 143.0, 353.0),
            (0.5, 15.0, 217.0, 545.0),
            (1.8, 35.0, 50.0, 100.0)
        ]
        
        for (β, T_dust, ν0, ν) in test_cases
            T_CMB = CMBForegrounds.T_CMB
            
            # Direct formula calculation
            r = ν / ν0
            expected = r^β * CMBForegrounds.Bnu_ratio(ν, ν0, T_dust) / CMBForegrounds.dBdT_ratio(ν, ν0, T_CMB)
            
            # Function result
            actual = CMBForegrounds.cib_mbb_sed_weight(β, T_dust, ν0, ν)
            
            @test actual ≈ expected rtol=1e-12
            @test actual > 0.0
        end
    end

    @testset "Mathematical Limit Verification" begin
        
        @testset "Power Law Dominance (r >> 1, β >> 0)" begin
            # When frequency ratio is large and β is large, r^β term dominates
            β_large = 3.0
            T_dust, ν0, ν = 25.0, 100.0, 1000.0  # r = 10
            
            weight = CMBForegrounds.cib_mbb_sed_weight(β_large, T_dust, ν0, ν)
            r = ν / ν0
            
            # The r^β component should dominate
            power_component = r^β_large  # = 10^3 = 1000
            @test weight > 100.0  # Should be large due to r^β
            
            # Compare with β = 0 case (no power law)
            weight_β0 = CMBForegrounds.cib_mbb_sed_weight(0.0, T_dust, ν0, ν)
            @test weight > 100 * weight_β0  # Much larger due to power law
        end
        
        @testset "Blackbody Dust Component Analysis" begin
            # Test different dust temperature regimes
            
            # Wien limit for dust (high frequency relative to dust temperature)
            T_dust_cold = 10.0   # Cold dust
            ν0, ν = 100.0, 1000.0  # High frequency
            β = 1.5
            
            weight_wien = CMBForegrounds.cib_mbb_sed_weight(β, T_dust_cold, ν0, ν)
            @test isfinite(weight_wien)
            @test weight_wien > 0.0
            
            # Compare with warmer dust at same frequency
            T_dust_warm = 30.0
            weight_warm = CMBForegrounds.cib_mbb_sed_weight(β, T_dust_warm, ν0, ν)
            
            # At high frequencies, warmer dust should generally be brighter
            @test weight_warm > weight_wien
        end
        
        @testset "CMB dBdT Component Analysis" begin
            # Test CMB temperature effects on normalization
            β, T_dust, ν0, ν = 1.5, 25.0, 143.0, 217.0
            
            # Test different CMB temperatures
            T_CMB_low = 2.0
            T_CMB_high = 4.0
            
            weight_low_CMB = CMBForegrounds.cib_mbb_sed_weight(β, T_dust, ν0, ν, T_CMB=T_CMB_low)
            weight_high_CMB = CMBForegrounds.cib_mbb_sed_weight(β, T_dust, ν0, ν, T_CMB=T_CMB_high)
            
            # Different CMB temperatures should give different normalizations
            @test weight_low_CMB != weight_high_CMB
            @test all([weight_low_CMB, weight_high_CMB] .> 0)
        end
        
        @testset "Frequency Ratio Scaling Analysis" begin
            # Test systematic frequency ratio effects
            β, T_dust = 1.5, 25.0
            ν0 = 143.0
            
            # Test different frequency ratios
            ratios = [0.5, 1.0, 2.0, 5.0, 10.0]
            frequencies = ratios .* ν0
            
            weights = [CMBForegrounds.cib_mbb_sed_weight(β, T_dust, ν0, ν) for ν in frequencies]
            
            # Reference frequency should give weight = 1
            @test weights[2] ≈ 1.0  # r = 1.0 case
            
            # Higher frequencies should generally give higher weights (CIB increases with frequency)
            @test weights[3] > weights[2]  # r = 2.0 > r = 1.0
            @test weights[4] > weights[3]  # r = 5.0 > r = 2.0
        end
    end

    @testset "Physics-Based Validation Tests" begin
        
        @testset "Realistic CIB Parameter Regimes" begin
            # Test with observationally motivated CIB parameters
            
            # Planck Collaboration parameters (typical CIB)
            β_planck = 1.6
            T_dust_planck = 24.0
            
            # Herschel-SPIRE frequencies for CIB studies
            herschel_freqs = [857.0, 1200.0, 2000.0]  # 350, 250, 150 μm
            reference_freq = 857.0
            
            weights = []
            for ν in herschel_freqs
                weight = CMBForegrounds.cib_mbb_sed_weight(β_planck, T_dust_planck, reference_freq, ν)
                push!(weights, weight)
                @test isfinite(weight)
                @test weight > 0
            end
            
            # Check expected CIB SED shape: increasing with frequency in submm
            @test weights[1] ≈ 1.0  # Reference frequency
            @test weights[2] > weights[1]  # Higher frequency
            @test weights[3] > weights[2]  # Even higher frequency
        end
        
        @testset "Galaxy Population Dust Properties" begin
            # Test different galaxy types with different dust properties
            
            # Local spirals: cooler dust
            β_spiral, T_dust_spiral = 1.2, 18.0
            
            # Starbursts: warmer dust
            β_starburst, T_dust_starburst = 1.8, 35.0
            
            # High-z dusty galaxies: intermediate
            β_highz, T_dust_highz = 1.5, 25.0
            
            ν0, ν = 143.0, 353.0  # Planck frequencies
            
            weight_spiral = CMBForegrounds.cib_mbb_sed_weight(β_spiral, T_dust_spiral, ν0, ν)
            weight_starburst = CMBForegrounds.cib_mbb_sed_weight(β_starburst, T_dust_starburst, ν0, ν)
            weight_highz = CMBForegrounds.cib_mbb_sed_weight(β_highz, T_dust_highz, ν0, ν)
            
            # All should be positive and finite
            @test all([weight_spiral, weight_starburst, weight_highz] .> 0)
            @test all(isfinite.([weight_spiral, weight_starburst, weight_highz]))
            
            # Different galaxy types should give different weights
            @test weight_spiral != weight_starburst != weight_highz
        end
        
        @testset "Multi-frequency CIB Analysis" begin
            # Simulate realistic multi-frequency CIB analysis
            β, T_dust = 1.6, 24.0
            
            # Planck + ground-based submm frequencies
            frequencies = [100.0, 143.0, 217.0, 353.0, 545.0, 857.0, 1200.0]
            reference = 353.0  # Common CIB reference
            
            weights = [CMBForegrounds.cib_mbb_sed_weight(β, T_dust, reference, ν) for ν in frequencies]
            
            # Reference frequency should be 1
            ref_idx = findfirst(x -> x == reference, frequencies)
            @test weights[ref_idx] ≈ 1.0
            
            # CIB should increase with frequency in submm
            for i in (ref_idx+1):length(weights)
                @test weights[i] > weights[i-1]
            end
            
            # Lower frequencies should be < 1
            for i in 1:(ref_idx-1)
                @test weights[i] < 1.0
            end
        end
        
        @testset "Dust Emissivity Index Effects" begin
            # Test range of β values from observations
            T_dust, ν0, ν = 25.0, 143.0, 545.0
            
            β_values = [0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5]  # Observational range
            weights = [CMBForegrounds.cib_mbb_sed_weight(β, T_dust, ν0, ν) for β in β_values]
            
            # Higher β should give higher weights at high frequencies
            for i in 2:length(weights)
                @test weights[i] > weights[i-1]
            end
            
            # All weights should be reasonable
            @test all(weights .> 0)
            @test all(weights .< 1e6)  # Allow large values for high β
        end
        
        @testset "Dust Temperature Effects on SED" begin
            # Test realistic dust temperature range
            β, ν0, ν = 1.5, 217.0, 857.0  # High frequency where temperature matters
            
            T_dust_values = [10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]  # K
            weights = [CMBForegrounds.cib_mbb_sed_weight(β, T_dust, ν0, ν) for T_dust in T_dust_values]
            
            # At high frequencies, warmer dust should be brighter
            for i in 2:length(weights)
                @test weights[i] > weights[i-1]
            end
            
            # Check physically reasonable ranges
            @test all(weights .> 0)
            @test all(isfinite.(weights))
        end
    end

    @testset "Extreme Value Numerical Stability" begin
        # Test with extreme parameter combinations
        extreme_cases = [
            (0.1, 5.0, 1000.0, 2000.0),    # Very low β, cold dust, high freq
            (5.0, 100.0, 10.0, 1000.0),     # Very high β, hot dust, large freq ratio
            (1.5, 25.0, 1e-3, 1e3),        # Extreme frequency ratio
            (1e-3, 1.0, 100.0, 200.0),     # Nearly zero β
            (1.5, 1000.0, 143.0, 217.0),   # Extremely hot dust
            (10.0, 25.0, 143.0, 145.0),    # Very high β, small freq ratio
        ]
        
        for (β, T_dust, ν0, ν) in extreme_cases
            weight = CMBForegrounds.cib_mbb_sed_weight(β, T_dust, ν0, ν)
            @test isfinite(weight)
            @test !isnan(weight)
            @test weight > 0
            @test weight < 1e20  # Avoid completely unreasonable values (allow for extreme cases)
        end
    end

    @testset "Performance and Advanced Edge Cases" begin
        Random.seed!(42)
        
        @testset "Random Parameter Stress Test" begin
            # Test 200 random parameter combinations
            for i in 1:200
                β = rand() * 4.0 + 0.1           # 0.1 to 4.1
                T_dust = rand() * 80.0 + 5.0     # 5 to 85 K
                ν0 = 10^(rand() * 3.0 + 1.0)     # 10 to 1000 GHz
                ν = 10^(rand() * 3.0 + 1.0)      # 10 to 1000 GHz
                
                weight = CMBForegrounds.cib_mbb_sed_weight(β, T_dust, ν0, ν)
                @test isfinite(weight)
                @test weight > 0
            end
        end
        
        @testset "Parameter Correlation Analysis" begin
            # Test parameter correlations and dependencies
            
            # β vs frequency ratio correlation
            β_values = [0.5, 1.0, 1.5, 2.0, 2.5]
            T_dust, ν0, ν = 25.0, 100.0, 300.0  # r = 3
            
            weights_β = [CMBForegrounds.cib_mbb_sed_weight(β, T_dust, ν0, ν) for β in β_values]
            
            # Should increase monotonically with β (since r > 1)
            for i in 2:length(weights_β)
                @test weights_β[i] > weights_β[i-1]
            end
            
            # Dust temperature vs frequency dependence
            T_values = [15.0, 20.0, 25.0, 30.0, 35.0]
            β = 1.5
            
            weights_T = [CMBForegrounds.cib_mbb_sed_weight(β, T_dust, ν0, ν) for T_dust in T_values]
            
            # Should increase with temperature at high frequencies
            for i in 2:length(weights_T)
                @test weights_T[i] > weights_T[i-1]
            end
        end
        
        @testset "High Precision Edge Cases" begin
            # Test cases requiring high numerical precision
            
            # Very close to reference frequency
            β, T_dust, ν0 = 1.5, 25.0, 143.0
            ν_close = 143.0 + 1e-10
            
            weight_close = CMBForegrounds.cib_mbb_sed_weight(β, T_dust, ν0, ν_close)
            @test weight_close ≈ 1.0 rtol=1e-6
            
            # Machine epsilon differences
            ν_eps = ν0 * (1 + eps(Float64))
            weight_eps = CMBForegrounds.cib_mbb_sed_weight(β, T_dust, ν0, ν_eps)
            @test isfinite(weight_eps)
            
            # β very close to integer values
            for β_int in [1.0, 2.0, 3.0]
                β_close = β_int + 1e-12
                weight = CMBForegrounds.cib_mbb_sed_weight(β_close, T_dust, ν0, 200.0)
                @test isfinite(weight)
            end
        end
        
        @testset "Cross-Validation with Component Functions" begin
            # Verify consistency with individual component functions
            
            test_cases = [
                (1.2, 18.0, 100.0, 300.0),
                (2.0, 40.0, 217.0, 857.0),
                (0.8, 12.0, 353.0, 545.0)
            ]
            
            for (β, T_dust, ν0, ν) in test_cases
                T_CMB = CMBForegrounds.T_CMB
                
                # Individual components
                r = ν / ν0
                power_component = r^β
                bnu_component = CMBForegrounds.Bnu_ratio(ν, ν0, T_dust)
                dbdt_component = CMBForegrounds.dBdT_ratio(ν, ν0, T_CMB)
                
                # Combined weight
                weight = CMBForegrounds.cib_mbb_sed_weight(β, T_dust, ν0, ν)
                expected = power_component * bnu_component / dbdt_component
                
                @test weight ≈ expected rtol=1e-10
                
                # Component reasonableness checks
                @test power_component > 0
                @test bnu_component > 0
                @test dbdt_component > 0
            end
        end
        
        @testset "Frequency Grid Consistency" begin
            # Test on regular frequency grids
            β, T_dust = 1.5, 25.0
            ν0 = 143.0
            
            # Linear grid
            freqs_lin = range(50.0, 1000.0, length=20)
            weights_lin = [CMBForegrounds.cib_mbb_sed_weight(β, T_dust, ν0, ν) for ν in freqs_lin]
            
            @test all(isfinite.(weights_lin))
            @test all(weights_lin .> 0)
            
            # Log grid (more realistic for SED fitting)
            freqs_log = 10 .^ range(log10(30.0), log10(3000.0), length=30)
            weights_log = [CMBForegrounds.cib_mbb_sed_weight(β, T_dust, ν0, ν) for ν in freqs_log]
            
            @test all(isfinite.(weights_log))
            @test all(weights_log .> 0)
            
            # Should be monotonically increasing in submm (most of the range)
            monotonic_region = freqs_log .> 200.0  # Above peak typically
            if any(monotonic_region)
                weights_mono = weights_log[monotonic_region]
                for i in 2:length(weights_mono)
                    @test weights_mono[i] >= weights_mono[i-1]  # Allow equal for numerical precision
                end
            end
        end
        
        @testset "Dimensional Analysis and Units" begin
            # Test dimensional consistency (weights should be dimensionless ratios)
            
            β, T_dust, ν0, ν = 1.5, 25.0, 143.0, 217.0
            
            # Basic weight
            weight = CMBForegrounds.cib_mbb_sed_weight(β, T_dust, ν0, ν)
            
            # Scale all frequencies by same factor (dimensionless result should change)
            freq_scale = 1.5
            weight_scaled = CMBForegrounds.cib_mbb_sed_weight(β, T_dust, ν0*freq_scale, ν*freq_scale)
            
            @test weight != weight_scaled  # Different because components have different temp dependencies
            
            # But both should be positive and finite
            @test weight > 0 && isfinite(weight)
            @test weight_scaled > 0 && isfinite(weight_scaled)
        end
        
        @testset "Robustness and Error Handling" begin
            # Test robustness to potential numerical issues
            
            β, T_dust, ν0, ν = 1.5, 25.0, 143.0, 217.0
            
            # Test with parameters that might cause issues in individual components
            
            # Very high frequency ratios
            weight_high_ratio = CMBForegrounds.cib_mbb_sed_weight(β, T_dust, 1.0, 10000.0)
            @test isfinite(weight_high_ratio)
            @test weight_high_ratio > 0
            
            # Very low frequency ratios  
            weight_low_ratio = CMBForegrounds.cib_mbb_sed_weight(β, T_dust, 1000.0, 1.0)
            @test isfinite(weight_low_ratio)
            @test weight_low_ratio > 0
            
            # Edge cases for blackbody functions
            # (These should be handled by the individual Bnu_ratio and dBdT_ratio functions)
            weight_edge = CMBForegrounds.cib_mbb_sed_weight(β, 0.1, ν0, ν)  # Very cold dust
            @test isfinite(weight_edge)
            @test weight_edge > 0
        end
    end
end
