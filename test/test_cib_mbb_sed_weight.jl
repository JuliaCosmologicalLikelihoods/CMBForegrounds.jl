"""
Unit tests for cib_mbb_sed_weight function

Tests the CIB (Cosmic Infrared Background) modified blackbody spectral energy distribution weight
Weight = r^β * B_ν(T_dust)/B_ν0(T_dust) / (dB/dT)_ν/(dB/dT)_ν0 (T_CMB)
"""

using Test
using CMBForegrounds

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
                                                 CMBForegrounds.dBdT_ratio(217.0, 100.0, CMBForegrounds.T_CMB)) rtol=1e-10
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
        @test weight ≈ 1.0 rtol=0.01  # Should be close to 1
        
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
end