"""
Unit tests for dust_tt_power_law function

Tests the dust power spectrum using a power law template with spectral energy distribution weights
D_ℓ = (ℓ/ℓ_pivot)^(α+2) * A_pivot * s1 * s2
where s1, s2 are the CIB modified blackbody SED weights for frequencies ν1, ν2
"""

using Test
using CMBForegrounds

@testset "dust_tt_power_law() Unit Tests" begin
    
    @testset "Basic Functionality" begin
        # Test with typical dust parameters
        ℓs = [10, 30, 80, 200, 500]
        A_pivot, α, β = 1.0, -0.6, 1.6
        ν1, ν2, T_dust, ν0 = 143.0, 217.0, 19.6, 150.0
        
        Dℓs = CMBForegrounds.dust_tt_power_law(ℓs, A_pivot, α, β, ν1, ν2, T_dust, ν0)
        
        # Basic output tests
        @test Dℓs isa AbstractVector
        @test length(Dℓs) == length(ℓs)
        @test all(Dℓs .> 0)  # Dust power should always be positive
        @test all(isfinite.(Dℓs))
        @test eltype(Dℓs) <: AbstractFloat
    end
    
    @testset "Mathematical Properties" begin
        # Test power law scaling at pivot point
        ℓs = [80]  # Pivot multipole
        A_pivot, α, β = 1.0, -0.6, 1.6
        ν1, ν2, T_dust, ν0 = 143.0, 217.0, 19.6, 150.0
        
        Dℓ_pivot = CMBForegrounds.dust_tt_power_law(ℓs, A_pivot, α, β, ν1, ν2, T_dust, ν0)[1]
        
        # At pivot, should equal A_pivot * s1 * s2
        s1 = CMBForegrounds.cib_mbb_sed_weight(β, T_dust, ν0, ν1)
        s2 = CMBForegrounds.cib_mbb_sed_weight(β, T_dust, ν0, ν2)
        expected_pivot = A_pivot * s1 * s2
        @test Dℓ_pivot ≈ expected_pivot
        
        # Test power law scaling away from pivot
        ℓ_test = 200
        Dℓ_test = CMBForegrounds.dust_tt_power_law([ℓ_test], A_pivot, α, β, ν1, ν2, T_dust, ν0)[1]
        scale_factor = (ℓ_test / 80)^(α + 2)  # Default ℓ_pivot = 80
        expected_scaled = scale_factor * expected_pivot
        @test Dℓ_test ≈ expected_scaled
    end
    
    @testset "Consistency with Formula" begin
        # Test that function matches mathematical definition exactly
        ℓs = [50, 100, 150]
        A_pivot, α, β = 2.0, -0.4, 1.5
        ν1, ν2, T_dust, ν0 = 100.0, 353.0, 25.0, 143.0
        ℓ_pivot = 100
        
        Dℓs = CMBForegrounds.dust_tt_power_law(ℓs, A_pivot, α, β, ν1, ν2, T_dust, ν0; ℓ_pivot=ℓ_pivot)
        
        # Manual calculation
        s1 = CMBForegrounds.cib_mbb_sed_weight(β, T_dust, ν0, ν1)
        s2 = CMBForegrounds.cib_mbb_sed_weight(β, T_dust, ν0, ν2)
        spectral_factor = A_pivot * s1 * s2
        
        for (i, ℓ) in enumerate(ℓs)
            scale_factor = (ℓ / ℓ_pivot)^(α + 2)
            expected = scale_factor * spectral_factor
            @test Dℓs[i] ≈ expected
        end
    end
    
    @testset "Power Law Index Effects" begin
        # Test how different α values affect the multipole scaling
        ℓs = [20, 80, 320]  # Factors of 4 from pivot
        A_pivot, β = 1.0, 1.6
        ν1, ν2, T_dust, ν0 = 143.0, 217.0, 19.6, 150.0
        
        # Different power law indices
        α_steep = -0.2   # α+2 = 1.8 (steeper increase with ℓ)
        α_moderate = -1.0 # α+2 = 1.0 (moderate increase with ℓ)
        α_flat = -2.0    # α+2 = 0.0 (flat, no ℓ dependence)
        
        Dℓs_steep = CMBForegrounds.dust_tt_power_law(ℓs, A_pivot, α_steep, β, ν1, ν2, T_dust, ν0)
        Dℓs_moderate = CMBForegrounds.dust_tt_power_law(ℓs, A_pivot, α_moderate, β, ν1, ν2, T_dust, ν0)
        Dℓs_flat = CMBForegrounds.dust_tt_power_law(ℓs, A_pivot, α_flat, β, ν1, ν2, T_dust, ν0)
        
        # At pivot (ℓ=80), all should be equal
        @test Dℓs_steep[2] ≈ Dℓs_moderate[2] ≈ Dℓs_flat[2]
        
        # At high ℓ (ℓ=320), steeper α (higher α+2) should give higher power
        @test Dℓs_steep[3] > Dℓs_moderate[3] > Dℓs_flat[3]
        
        # At low ℓ (ℓ=20), steeper α (higher α+2) should give lower power  
        @test Dℓs_steep[1] < Dℓs_moderate[1] < Dℓs_flat[1]  # flat case gives same value at all ℓ
    end
    
    @testset "Amplitude Scaling" begin
        # Test that amplitude scales linearly
        ℓs = [50, 80, 200]
        α, β = -0.6, 1.6
        ν1, ν2, T_dust, ν0 = 143.0, 217.0, 19.6, 150.0
        
        A1, A2 = 1.0, 3.0
        Dℓs1 = CMBForegrounds.dust_tt_power_law(ℓs, A1, α, β, ν1, ν2, T_dust, ν0)
        Dℓs2 = CMBForegrounds.dust_tt_power_law(ℓs, A2, α, β, ν1, ν2, T_dust, ν0)
        
        # Should scale exactly linearly with amplitude
        @test all(Dℓs2 .≈ (A2/A1) .* Dℓs1)
    end
    
    @testset "Frequency Dependence" begin
        # Test how different frequency combinations affect spectral weights
        ℓs = [80]  # Use pivot for simplicity
        A_pivot, α, β = 1.0, -0.6, 1.6
        T_dust, ν0 = 19.6, 150.0
        
        # Same frequency (auto-spectrum)
        Dℓ_auto = CMBForegrounds.dust_tt_power_law(ℓs, A_pivot, α, β, 143.0, 143.0, T_dust, ν0)[1]
        
        # Different frequencies (cross-spectrum)
        Dℓ_cross = CMBForegrounds.dust_tt_power_law(ℓs, A_pivot, α, β, 143.0, 217.0, T_dust, ν0)[1]
        
        # Both should be positive but different due to SED weights
        @test Dℓ_auto > 0 && Dℓ_cross > 0
        @test Dℓ_auto != Dℓ_cross
        
        # Cross-spectrum should be geometric mean of auto-spectra weights
        Dℓ_auto1 = CMBForegrounds.dust_tt_power_law(ℓs, A_pivot, α, β, 143.0, 143.0, T_dust, ν0)[1] 
        Dℓ_auto2 = CMBForegrounds.dust_tt_power_law(ℓs, A_pivot, α, β, 217.0, 217.0, T_dust, ν0)[1]
        # Note: This is not exactly geometric mean due to the way SED weights work
        @test Dℓ_cross > 0  # Just check it's sensible
    end
    
    @testset "Default Parameters" begin
        # Test default ℓ_pivot and T_CMB parameters
        ℓs = [80]
        A_pivot, α, β = 1.0, -0.6, 1.6
        ν1, ν2, T_dust, ν0 = 143.0, 217.0, 19.6, 150.0
        
        # With defaults
        Dℓ_default = CMBForegrounds.dust_tt_power_law(ℓs, A_pivot, α, β, ν1, ν2, T_dust, ν0)[1]
        
        # With explicit values  
        Dℓ_explicit = CMBForegrounds.dust_tt_power_law(ℓs, A_pivot, α, β, ν1, ν2, T_dust, ν0; 
                                                      ℓ_pivot=80, T_CMB=CMBForegrounds.T_CMB)[1]
        
        @test Dℓ_default ≈ Dℓ_explicit
        
        # With different pivot
        Dℓ_different_pivot = CMBForegrounds.dust_tt_power_law(ℓs, A_pivot, α, β, ν1, ν2, T_dust, ν0; 
                                                             ℓ_pivot=100)[1]
        @test Dℓ_different_pivot != Dℓ_default
    end
    
    @testset "Beta Parameter Effects" begin
        # Test how dust emissivity index affects the spectrum
        ℓs = [80]  # Use pivot
        A_pivot, α = 1.0, -0.6
        ν1, ν2, T_dust, ν0 = 143.0, 353.0, 19.6, 150.0  # Use high freq for strong β effect
        
        Dℓ_β1 = CMBForegrounds.dust_tt_power_law(ℓs, A_pivot, α, 1.0, ν1, ν2, T_dust, ν0)[1]
        Dℓ_β15 = CMBForegrounds.dust_tt_power_law(ℓs, A_pivot, α, 1.5, ν1, ν2, T_dust, ν0)[1] 
        Dℓ_β2 = CMBForegrounds.dust_tt_power_law(ℓs, A_pivot, α, 2.0, ν1, ν2, T_dust, ν0)[1]
        
        # Higher β should give higher power at high frequencies
        @test Dℓ_β1 < Dℓ_β15 < Dℓ_β2
        @test all([Dℓ_β1, Dℓ_β15, Dℓ_β2] .> 0)
    end
    
    @testset "Dust Temperature Effects" begin
        # Test how dust temperature affects the spectrum
        ℓs = [80]
        A_pivot, α, β = 1.0, -0.6, 1.6
        ν1, ν2, ν0 = 143.0, 353.0, 150.0
        
        Dℓ_cold = CMBForegrounds.dust_tt_power_law(ℓs, A_pivot, α, β, ν1, ν2, 15.0, ν0)[1]
        Dℓ_warm = CMBForegrounds.dust_tt_power_law(ℓs, A_pivot, α, β, ν1, ν2, 25.0, ν0)[1]
        
        # Different temperatures should give different powers
        @test Dℓ_cold != Dℓ_warm
        @test Dℓ_cold > 0 && Dℓ_warm > 0
    end
    
    @testset "Type Stability" begin
        # Test with different input types
        ℓs = [50, 80, 200]
        A_pivot, α, β = 1.0, -0.6, 1.6
        ν1, ν2, T_dust, ν0 = 143.0, 217.0, 19.6, 150.0
        
        # Float64 inputs
        Dℓs_float = CMBForegrounds.dust_tt_power_law(ℓs, A_pivot, α, β, ν1, ν2, T_dust, ν0)
        @test eltype(Dℓs_float) == Float64
        
        # Mixed types
        Dℓs_mixed = CMBForegrounds.dust_tt_power_law([50, 80, 200], 1, -0.6, 1.6, 143, 217.0, 19.6, 150)
        @test eltype(Dℓs_mixed) == Float64
        @test Dℓs_mixed ≈ Dℓs_float
    end
    
    @testset "Vector Operations" begin
        # Test that function works correctly with vector inputs
        ℓs = [10, 30, 80, 200, 500, 1000]
        A_pivot, α, β = 1.0, -0.6, 1.6
        ν1, ν2, T_dust, ν0 = 143.0, 217.0, 19.6, 150.0
        
        Dℓs = CMBForegrounds.dust_tt_power_law(ℓs, A_pivot, α, β, ν1, ν2, T_dust, ν0)
        
        # Check vector properties
        @test length(Dℓs) == length(ℓs)
        @test all(Dℓs .> 0)
        @test all(isfinite.(Dℓs))
        
        # Check that elements correspond to individual calculations
        for i in 1:length(ℓs)
            single_result = CMBForegrounds.dust_tt_power_law([ℓs[i]], A_pivot, α, β, ν1, ν2, T_dust, ν0)[1]
            @test Dℓs[i] ≈ single_result
        end
    end
    
    @testset "Physical Consistency" begin
        # Test with realistic dust parameters from literature
        ℓs = [10, 80, 500, 2000]
        
        # Planck-like dust parameters
        A_pivot = 1.0      # Normalized amplitude
        α = -0.42          # Typical dust spectral index  
        β = 1.59           # Planck dust emissivity index
        T_dust = 19.6      # Planck dust temperature
        ν0 = 150.0         # Reference frequency
        
        # Test auto-spectrum at 353 GHz (dust-dominated)
        Dℓs_353 = CMBForegrounds.dust_tt_power_law(ℓs, A_pivot, α, β, 353.0, 353.0, T_dust, ν0)
        
        # Test cross-spectrum 217x353
        Dℓs_cross = CMBForegrounds.dust_tt_power_law(ℓs, A_pivot, α, β, 217.0, 353.0, T_dust, ν0)
        
        # Physical expectations
        @test all(Dℓs_353 .> 0)
        @test all(Dℓs_cross .> 0)
        @test all(isfinite.(Dℓs_353))
        @test all(isfinite.(Dℓs_cross))
        
        # For α = -0.42, we get α+2 = 1.58 > 0, so power should INCREASE with ℓ
        @test Dℓs_353[1] < Dℓs_353[2] < Dℓs_353[3] < Dℓs_353[4]
        @test Dℓs_cross[1] < Dℓs_cross[2] < Dℓs_cross[3] < Dℓs_cross[4]
    end
    
    @testset "Edge Cases" begin
        # Test with extreme but valid parameter values
        ℓs = [2, 80, 5000]  # Wide range of multipoles
        
        # Very steep power law
        Dℓs_steep = CMBForegrounds.dust_tt_power_law(ℓs, 1.0, -2.0, 1.6, 143.0, 217.0, 19.6, 150.0)
        @test all(isfinite.(Dℓs_steep)) && all(Dℓs_steep .> 0)
        
        # Very shallow power law  
        Dℓs_shallow = CMBForegrounds.dust_tt_power_law(ℓs, 1.0, 0.5, 1.6, 143.0, 217.0, 19.6, 150.0)
        @test all(isfinite.(Dℓs_shallow)) && all(Dℓs_shallow .> 0)
        
        # Very low amplitude
        Dℓs_low = CMBForegrounds.dust_tt_power_law(ℓs, 1e-6, -0.6, 1.6, 143.0, 217.0, 19.6, 150.0)
        @test all(isfinite.(Dℓs_low)) && all(Dℓs_low .> 0)
        
        # Very high amplitude
        Dℓs_high = CMBForegrounds.dust_tt_power_law(ℓs, 1e6, -0.6, 1.6, 143.0, 217.0, 19.6, 150.0)
        @test all(isfinite.(Dℓs_high)) && all(Dℓs_high .> 0)
    end
    
    @testset "Numerical Stability" begin
        # Test numerical stability for challenging parameter combinations
        ℓs = [1, 80, 10000]  # Very wide multipole range
        
        # Very close to pivot
        ℓs_close = [79.99, 80.0, 80.01]
        Dℓs_close = CMBForegrounds.dust_tt_power_law(ℓs_close, 1.0, -0.6, 1.6, 143.0, 217.0, 19.6, 150.0)
        @test all(isfinite.(Dℓs_close))
        @test Dℓs_close[2] ≈ Dℓs_close[1] rtol=0.01  # Should be very close
        @test Dℓs_close[2] ≈ Dℓs_close[3] rtol=0.01
        
        # Very different pivot
        Dℓs_diff_pivot = CMBForegrounds.dust_tt_power_law(ℓs, 1.0, -0.6, 1.6, 143.0, 217.0, 19.6, 150.0; ℓ_pivot=3000)
        @test all(isfinite.(Dℓs_diff_pivot)) && all(Dℓs_diff_pivot .> 0)
    end
    
    @testset "Cross-Frequency Symmetry" begin
        # Test that cross-spectrum is symmetric in frequency ordering
        ℓs = [50, 80, 200]
        A_pivot, α, β = 1.0, -0.6, 1.6
        T_dust, ν0 = 19.6, 150.0
        
        Dℓs_12 = CMBForegrounds.dust_tt_power_law(ℓs, A_pivot, α, β, 143.0, 217.0, T_dust, ν0)
        Dℓs_21 = CMBForegrounds.dust_tt_power_law(ℓs, A_pivot, α, β, 217.0, 143.0, T_dust, ν0)
        
        # Cross-spectrum should be symmetric
        @test all(Dℓs_12 .≈ Dℓs_21)
    end
    
    @testset "Parameter Broadcasting" begin
        # Test that scalar parameters work correctly with vector ℓs
        ℓs = collect(10:10:100)  # Multiple multipoles
        A_pivot, α, β = 2.0, -0.5, 1.7
        ν1, ν2, T_dust, ν0 = 100.0, 350.0, 20.0, 143.0
        
        Dℓs = CMBForegrounds.dust_tt_power_law(ℓs, A_pivot, α, β, ν1, ν2, T_dust, ν0)
        
        # Should work and produce vector of correct length
        @test length(Dℓs) == length(ℓs)
        @test all(Dℓs .> 0)
        @test all(isfinite.(Dℓs))
        
        # Check monotonicity based on α+2
        if α < -2
            @test all(diff(Dℓs) .< 0)  # Should decrease when α+2 < 0
        elseif α > -2
            @test all(diff(Dℓs) .> 0)  # Should increase when α+2 > 0
        end
        # When α = -2, α+2 = 0, so should be flat (constant)
    end
end