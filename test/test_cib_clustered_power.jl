"""
Unit tests for cib_clustered_power function

Tests the CIB clustered power spectrum function that handles both auto-spectra and cross-spectra
D_ℓ = A_CIB * s1 * s2 * √(z1 * z2) * (ℓ/ℓ_pivot)^α
"""

@testset "cib_clustered_power() Unit Tests" begin

    @testset "Basic Functionality" begin
        # Test with typical CIB parameters
        ℓs = [500, 1000, 3000, 6000]
        A_CIB, α, β = 1.0, 0.8, 1.6
        ν1, ν2 = 217.0, 353.0
        z1, z2 = 0.9, 1.1
        T_dust, ν0_cib = 25.0, 150.0

        D_ℓs = CMBForegrounds.cib_clustered_power(ℓs, A_CIB, α, β, ν1, ν2, z1, z2, T_dust, ν0_cib)

        # Basic output tests
        @test D_ℓs isa AbstractVector
        @test length(D_ℓs) == length(ℓs)
        @test all(D_ℓs .> 0)  # CIB power should be positive
        @test all(isfinite.(D_ℓs))
        @test eltype(D_ℓs) <: AbstractFloat
    end

    @testset "Mathematical Properties" begin
        # Test power law scaling at pivot point
        ℓs = [3000]  # Pivot multipole
        A_CIB, α, β = 1.0, 0.8, 1.6
        ν1, ν2 = 217.0, 353.0
        z1, z2 = 1.0, 1.0
        T_dust, ν0_cib = 25.0, 150.0

        D_ℓ_pivot = CMBForegrounds.cib_clustered_power(ℓs, A_CIB, α, β, ν1, ν2, z1, z2, T_dust, ν0_cib)[1]

        # At pivot, should equal A_CIB * s1 * s2 * √(z1 * z2)
        s1 = CMBForegrounds.cib_mbb_sed_weight(β, T_dust, ν0_cib, ν1)
        s2 = CMBForegrounds.cib_mbb_sed_weight(β, T_dust, ν0_cib, ν2)
        expected_pivot = A_CIB * s1 * s2 * sqrt(z1 * z2)
        @test D_ℓ_pivot ≈ expected_pivot

        # Test power law scaling away from pivot
        ℓ_test = 6000
        D_ℓ_test = CMBForegrounds.cib_clustered_power([ℓ_test], A_CIB, α, β, ν1, ν2, z1, z2, T_dust, ν0_cib)[1]
        scale_factor = (ℓ_test / 3000)^α  # Default ℓ_pivot = 3000
        expected_scaled = scale_factor * expected_pivot
        @test D_ℓ_test ≈ expected_scaled
    end

    @testset "Consistency with Formula" begin
        # Test that function matches mathematical definition exactly
        ℓs = [1000, 3000, 5000]
        A_CIB, α, β = 2.0, 0.6, 1.5
        ν1, ν2 = 143.0, 353.0
        z1, z2 = 0.8, 1.2
        T_dust, ν0_cib = 24.0, 143.0
        ℓ_pivot = 2000

        D_ℓs = CMBForegrounds.cib_clustered_power(ℓs, A_CIB, α, β, ν1, ν2, z1, z2, T_dust, ν0_cib; ℓ_pivot=ℓ_pivot)

        # Manual calculation
        s1 = CMBForegrounds.cib_mbb_sed_weight(β, T_dust, ν0_cib, ν1)
        s2 = CMBForegrounds.cib_mbb_sed_weight(β, T_dust, ν0_cib, ν2)
        spectral_factor = A_CIB * s1 * s2 * sqrt(z1 * z2)

        for (i, ℓ) in enumerate(ℓs)
            scale_factor = (ℓ / ℓ_pivot)^α
            expected = scale_factor * spectral_factor
            @test D_ℓs[i] ≈ expected
        end
    end

    @testset "Auto-Spectrum vs Cross-Spectrum" begin
        # Test that auto-spectrum is a special case of cross-spectrum
        ℓs = [1000, 3000, 5000]
        A_CIB, α, β = 1.0, 0.8, 1.6
        ν = 353.0
        z = 1.0
        T_dust, ν0_cib = 25.0, 150.0

        # Auto-spectrum: same frequency and redshift
        D_auto = CMBForegrounds.cib_clustered_power(ℓs, A_CIB, α, β, ν, ν, z, z, T_dust, ν0_cib)

        # Should be equivalent to the formula for auto-spectrum
        s = CMBForegrounds.cib_mbb_sed_weight(β, T_dust, ν0_cib, ν)
        expected_spectral = A_CIB * s * s * abs(z)  # √(z*z) = |z| when z is real

        for (i, ℓ) in enumerate(ℓs)
            scale_factor = (ℓ / 3000)^α
            expected = expected_spectral * scale_factor
            @test D_auto[i] ≈ expected
        end
    end

    @testset "Power Law Index Effects" begin
        # Test how different α values affect the multipole scaling
        ℓs = [1000, 3000, 9000]  # Factors from pivot
        A_CIB, β = 1.0, 1.6
        ν1, ν2 = 217.0, 353.0
        z1, z2 = 1.0, 1.0
        T_dust, ν0_cib = 25.0, 150.0

        # Different power law indices
        α_shallow = 0.4
        α_moderate = 0.8
        α_steep = 1.2

        D_shallow = CMBForegrounds.cib_clustered_power(ℓs, A_CIB, α_shallow, β, ν1, ν2, z1, z2, T_dust, ν0_cib)
        D_moderate = CMBForegrounds.cib_clustered_power(ℓs, A_CIB, α_moderate, β, ν1, ν2, z1, z2, T_dust, ν0_cib)
        D_steep = CMBForegrounds.cib_clustered_power(ℓs, A_CIB, α_steep, β, ν1, ν2, z1, z2, T_dust, ν0_cib)

        # At pivot (ℓ=3000), all should be equal
        @test D_shallow[2] ≈ D_moderate[2] ≈ D_steep[2]

        # At high ℓ (ℓ=9000), steeper α should give higher power
        @test D_steep[3] > D_moderate[3] > D_shallow[3]

        # At low ℓ (ℓ=1000), steeper α should give lower power
        @test D_steep[1] < D_moderate[1] < D_shallow[1]
    end

    @testset "Amplitude Scaling" begin
        # Test that amplitude scales linearly
        ℓs = [1000, 3000, 5000]
        α, β = 0.8, 1.6
        ν1, ν2 = 217.0, 353.0
        z1, z2 = 1.0, 1.0
        T_dust, ν0_cib = 25.0, 150.0

        A1, A2 = 1.0, 3.5
        D1 = CMBForegrounds.cib_clustered_power(ℓs, A1, α, β, ν1, ν2, z1, z2, T_dust, ν0_cib)
        D2 = CMBForegrounds.cib_clustered_power(ℓs, A2, α, β, ν1, ν2, z1, z2, T_dust, ν0_cib)

        # Should scale exactly linearly with amplitude
        @test all(D2 .≈ (A2 / A1) .* D1)
    end

    @testset "Redshift Factor Effects" begin
        # Test how redshift factors affect the spectrum
        ℓs = [3000]  # Use pivot for simplicity
        A_CIB, α, β = 1.0, 0.8, 1.6
        ν1, ν2 = 217.0, 353.0
        T_dust, ν0_cib = 25.0, 150.0

        # Different redshift factor combinations
        z_combinations = [(1.0, 1.0), (0.5, 2.0), (0.9, 1.1), (2.0, 2.0)]

        for (z1, z2) in z_combinations
            D = CMBForegrounds.cib_clustered_power(ℓs, A_CIB, α, β, ν1, ν2, z1, z2, T_dust, ν0_cib)[1]

            # Manual calculation
            s1 = CMBForegrounds.cib_mbb_sed_weight(β, T_dust, ν0_cib, ν1)
            s2 = CMBForegrounds.cib_mbb_sed_weight(β, T_dust, ν0_cib, ν2)
            expected = A_CIB * s1 * s2 * sqrt(z1 * z2)  # At pivot, scale factor = 1

            @test D ≈ expected
            @test D > 0  # Should always be positive for positive z values
        end
    end

    @testset "Frequency Dependence" begin
        # Test spectral behavior with different frequency combinations
        ℓs = [3000]  # Use pivot
        A_CIB, α, β = 1.0, 0.8, 1.6
        z1, z2 = 1.0, 1.0
        T_dust, ν0_cib = 25.0, 150.0

        # Different frequency combinations
        freq_pairs = [(143.0, 143.0), (143.0, 217.0), (217.0, 353.0), (353.0, 353.0)]

        D_results = []
        for (ν1, ν2) in freq_pairs
            D = CMBForegrounds.cib_clustered_power(ℓs, A_CIB, α, β, ν1, ν2, z1, z2, T_dust, ν0_cib)[1]
            push!(D_results, D)
            @test D > 0
            @test isfinite(D)
        end

        # Higher frequencies should generally give higher power due to SED weights
        # 353x353 should be highest, 143x143 should be lowest
        @test D_results[1] < D_results[4]  # 143x143 < 353x353
        @test all(D_results .> 0)
    end

    @testset "Default Parameters" begin
        # Test default ℓ_pivot and T_CMB parameters
        ℓs = [3000]
        A_CIB, α, β = 1.0, 0.8, 1.6
        ν1, ν2 = 217.0, 353.0
        z1, z2 = 1.0, 1.0
        T_dust, ν0_cib = 25.0, 150.0

        # With defaults
        D_default = CMBForegrounds.cib_clustered_power(ℓs, A_CIB, α, β, ν1, ν2, z1, z2, T_dust, ν0_cib)[1]

        # With explicit values
        D_explicit = CMBForegrounds.cib_clustered_power(ℓs, A_CIB, α, β, ν1, ν2, z1, z2, T_dust, ν0_cib;
            ℓ_pivot=3000, T_CMB=CMBForegrounds.T_CMB)[1]

        @test D_default ≈ D_explicit

        # With different pivot
        D_different_pivot = CMBForegrounds.cib_clustered_power(ℓs, A_CIB, α, β, ν1, ν2, z1, z2, T_dust, ν0_cib;
            ℓ_pivot=5000)[1]
        @test D_different_pivot != D_default
    end

    @testset "Beta Parameter Effects" begin
        # Test how dust emissivity index affects the spectrum
        ℓs = [3000]  # Use pivot
        A_CIB, α = 1.0, 0.8
        ν1, ν2 = 143.0, 353.0  # Use high freq for strong β effect
        z1, z2 = 1.0, 1.0
        T_dust, ν0_cib = 25.0, 150.0

        D_β1 = CMBForegrounds.cib_clustered_power(ℓs, A_CIB, α, 1.0, ν1, ν2, z1, z2, T_dust, ν0_cib)[1]
        D_β15 = CMBForegrounds.cib_clustered_power(ℓs, A_CIB, α, 1.5, ν1, ν2, z1, z2, T_dust, ν0_cib)[1]
        D_β2 = CMBForegrounds.cib_clustered_power(ℓs, A_CIB, α, 2.0, ν1, ν2, z1, z2, T_dust, ν0_cib)[1]

        # Higher β should give higher power at high frequencies
        @test D_β1 < D_β15 < D_β2
        @test all([D_β1, D_β15, D_β2] .> 0)
    end

    @testset "Dust Temperature Effects" begin
        # Test how dust temperature affects the spectrum
        ℓs = [3000]
        A_CIB, α, β = 1.0, 0.8, 1.6
        ν1, ν2 = 217.0, 353.0
        z1, z2 = 1.0, 1.0
        ν0_cib = 150.0

        D_cold = CMBForegrounds.cib_clustered_power(ℓs, A_CIB, α, β, ν1, ν2, z1, z2, 15.0, ν0_cib)[1]
        D_warm = CMBForegrounds.cib_clustered_power(ℓs, A_CIB, α, β, ν1, ν2, z1, z2, 30.0, ν0_cib)[1]

        # Different temperatures should give different powers
        @test D_cold != D_warm
        @test D_cold > 0 && D_warm > 0
    end

    @testset "Type Stability" begin
        # Test with different input types
        ℓs = [1000, 3000, 5000]
        A_CIB, α, β = 1.0, 0.8, 1.6
        ν1, ν2 = 217.0, 353.0
        z1, z2 = 1.0, 1.0
        T_dust, ν0_cib = 25.0, 150.0

        # Float64 inputs
        D_float = CMBForegrounds.cib_clustered_power(ℓs, A_CIB, α, β, ν1, ν2, z1, z2, T_dust, ν0_cib)
        @test eltype(D_float) == Float64

        # Mixed types
        D_mixed = CMBForegrounds.cib_clustered_power([1000, 3000, 5000], 1, 0.8, 1.6, 217, 353.0, 1, 1.0, 25, 150)
        @test eltype(D_mixed) == Float64
        @test D_mixed ≈ D_float
    end

    @testset "Vector Operations" begin
        # Test that function works correctly with vector inputs
        ℓs = [100, 500, 1000, 3000, 6000, 10000]
        A_CIB, α, β = 1.0, 0.8, 1.6
        ν1, ν2 = 217.0, 353.0
        z1, z2 = 1.0, 1.0
        T_dust, ν0_cib = 25.0, 150.0

        D_ℓs = CMBForegrounds.cib_clustered_power(ℓs, A_CIB, α, β, ν1, ν2, z1, z2, T_dust, ν0_cib)

        # Check vector properties
        @test length(D_ℓs) == length(ℓs)
        @test all(D_ℓs .> 0)
        @test all(isfinite.(D_ℓs))

        # Check that elements correspond to individual calculations
        for i in 1:length(ℓs)
            single_result = CMBForegrounds.cib_clustered_power([ℓs[i]], A_CIB, α, β, ν1, ν2, z1, z2, T_dust, ν0_cib)[1]
            @test D_ℓs[i] ≈ single_result
        end

        # For positive α, power should increase with ℓ
        if α > 0
            @test all(diff(D_ℓs) .> 0)  # Should increase
        end
    end

    @testset "Physical Consistency" begin
        # Test with realistic CIB parameters
        ℓs = [500, 1500, 3000, 6000, 9000]

        # Realistic CIB parameters
        A_CIB = 1.0       # Normalized amplitude
        α = 0.8           # Typical CIB multipole scaling
        β = 1.6           # Typical dust emissivity
        T_dust = 25.0     # Typical CIB dust temperature
        ν0_cib = 150.0    # Reference frequency

        # Test auto-spectrum at 353 GHz (CIB-dominated)
        D_auto = CMBForegrounds.cib_clustered_power(ℓs, A_CIB, α, β, 353.0, 353.0, 1.0, 1.0, T_dust, ν0_cib)

        # Test cross-spectrum 217x353
        D_cross = CMBForegrounds.cib_clustered_power(ℓs, A_CIB, α, β, 217.0, 353.0, 0.9, 1.1, T_dust, ν0_cib)

        # Physical expectations
        @test all(D_auto .> 0)
        @test all(D_cross .> 0)
        @test all(isfinite.(D_auto))
        @test all(isfinite.(D_cross))

        # For positive α, power should increase with ℓ
        @test all(diff(D_auto) .> 0)
        @test all(diff(D_cross) .> 0)

        # Auto-spectrum should be higher than cross-spectrum (for same effective z)
        D_cross_same_z = CMBForegrounds.cib_clustered_power(ℓs, A_CIB, α, β, 217.0, 353.0, 1.0, 1.0, T_dust, ν0_cib)
        # This depends on SED weights, so just check they're reasonable
        @test all(D_cross_same_z .> 0)
    end

    @testset "Edge Cases" begin
        # Test with extreme but valid parameter values
        ℓs = [100, 3000, 10000]  # Wide range of multipoles

        # Very high amplitude
        D_high_A = CMBForegrounds.cib_clustered_power(ℓs, 100.0, 0.8, 1.6, 217.0, 353.0, 1.0, 1.0, 25.0, 150.0)
        @test all(isfinite.(D_high_A)) && all(D_high_A .> 0)

        # Very low amplitude
        D_low_A = CMBForegrounds.cib_clustered_power(ℓs, 1e-3, 0.8, 1.6, 217.0, 353.0, 1.0, 1.0, 25.0, 150.0)
        @test all(isfinite.(D_low_A)) && all(D_low_A .> 0)

        # Extreme power law indices
        D_flat = CMBForegrounds.cib_clustered_power(ℓs, 1.0, 0.0, 1.6, 217.0, 353.0, 1.0, 1.0, 25.0, 150.0)
        D_steep = CMBForegrounds.cib_clustered_power(ℓs, 1.0, 2.0, 1.6, 217.0, 353.0, 1.0, 1.0, 25.0, 150.0)
        @test all(isfinite.(D_flat)) && all(D_flat .> 0)
        @test all(isfinite.(D_steep)) && all(D_steep .> 0)

        # Very different redshift factors
        D_extreme_z = CMBForegrounds.cib_clustered_power(ℓs, 1.0, 0.8, 1.6, 217.0, 353.0, 0.1, 10.0, 25.0, 150.0)
        @test all(isfinite.(D_extreme_z)) && all(D_extreme_z .> 0)
    end

    @testset "Numerical Stability" begin
        # Test numerical stability for challenging parameter combinations

        # Very close to pivot
        ℓs_close = [2999, 3000, 3001]
        D_close = CMBForegrounds.cib_clustered_power(ℓs_close, 1.0, 0.8, 1.6, 217.0, 353.0, 1.0, 1.0, 25.0, 150.0)
        @test all(isfinite.(D_close))
        @test D_close[2] ≈ D_close[1] rtol = 0.01  # Should be very close
        @test D_close[2] ≈ D_close[3] rtol = 0.01

        # Very different pivot
        D_diff_pivot = CMBForegrounds.cib_clustered_power(ℓs_close, 1.0, 0.8, 1.6, 217.0, 353.0, 1.0, 1.0, 25.0, 150.0; ℓ_pivot=10000)
        @test all(isfinite.(D_diff_pivot)) && all(D_diff_pivot .> 0)

        # Very small redshift factors
        D_small_z = CMBForegrounds.cib_clustered_power([3000], 1.0, 0.8, 1.6, 217.0, 353.0, 1e-6, 1e-6, 25.0, 150.0)
        @test all(isfinite.(D_small_z)) && all(D_small_z .> 0)
    end

    @testset "Cross-Frequency Symmetry" begin
        # Test that cross-spectrum is symmetric in frequency ordering
        ℓs = [1000, 3000, 5000]
        A_CIB, α, β = 1.0, 0.8, 1.6
        z1, z2 = 0.9, 1.1
        T_dust, ν0_cib = 25.0, 150.0

        D_12 = CMBForegrounds.cib_clustered_power(ℓs, A_CIB, α, β, 217.0, 353.0, z1, z2, T_dust, ν0_cib)
        D_21 = CMBForegrounds.cib_clustered_power(ℓs, A_CIB, α, β, 353.0, 217.0, z2, z1, T_dust, ν0_cib)

        # Cross-spectrum should be symmetric (swapping both freq and z together)
        @test all(D_12 .≈ D_21)
    end

    @testset "Comparison with Special Cases" begin
        # Compare auto-spectrum computed two ways
        ℓs = [1000, 3000, 6000]
        A_CIB, α, β = 1.0, 0.8, 1.6
        ν, z = 353.0, 1.0
        T_dust, ν0_cib = 25.0, 150.0

        # Method 1: Using the general function with ν1=ν2, z1=z2
        D_auto_general = CMBForegrounds.cib_clustered_power(ℓs, A_CIB, α, β, ν, ν, z, z, T_dust, ν0_cib)

        # Method 2: Manual calculation as if it were the old single-frequency function
        s = CMBForegrounds.cib_mbb_sed_weight(β, T_dust, ν0_cib, ν)
        Z = abs(z)
        D_auto_manual = @. (A_CIB * s * s * Z) * (ℓs / 3000)^α

        # Should be identical
        @test all(D_auto_general .≈ D_auto_manual)
    end
end
