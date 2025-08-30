"""
Unit tests for tsz_cross_power function

Tests the tSZ cross-power spectrum function that computes:
D_ℓ = template * A_tSZ * g(ν1) * g(ν2) * (ℓ/ℓ_pivot)^α_tSZ
"""

@testset "tsz_cross_power() Unit Tests" begin

    @testset "Basic Functionality" begin
        # Test with typical tSZ parameters
        template = [10.0, 20.0, 50.0, 80.0]  # Arbitrary template shape
        A_tSZ = 1.0
        ν1, ν2 = 143.0, 217.0
        ν0 = 143.0
        α_tSZ = 0.8
        ℓ_pivot = 3000
        ℓs = [1000, 2000, 3000, 4000]

        D_ℓs = CMBForegrounds.tsz_cross_power(template, A_tSZ, ν1, ν2, ν0, α_tSZ, ℓ_pivot, ℓs)

        # Basic output tests
        @test D_ℓs isa AbstractVector
        @test length(D_ℓs) == length(template)
        @test length(D_ℓs) == length(ℓs)
        @test all(isfinite.(D_ℓs))
        @test eltype(D_ℓs) <: AbstractFloat
    end

    @testset "Mathematical Properties" begin
        # Test the mathematical formula: template * A_tSZ * g1 * g2 * (ℓ/ℓ_pivot)^α
        template = [1.0, 2.0, 3.0]
        A_tSZ = 2.0
        ν1, ν2 = 143.0, 217.0
        ν0 = 143.0
        α_tSZ = 0.5
        ℓ_pivot = 3000
        ℓs = [1500, 3000, 6000]

        D_ℓs = CMBForegrounds.tsz_cross_power(template, A_tSZ, ν1, ν2, ν0, α_tSZ, ℓ_pivot, ℓs)

        # Manual calculation
        g1 = CMBForegrounds.tsz_g_ratio(ν1, ν0, CMBForegrounds.T_CMB)
        g2 = CMBForegrounds.tsz_g_ratio(ν2, ν0, CMBForegrounds.T_CMB)
        spectral_factor = A_tSZ * g1 * g2

        for i in 1:length(ℓs)
            scale_factor = (ℓs[i] / ℓ_pivot)^α_tSZ
            expected = template[i] * spectral_factor * scale_factor
            @test D_ℓs[i] ≈ expected
        end
    end

    @testset "Template Scaling" begin
        # Test that output scales linearly with template
        template1 = [1.0, 2.0, 3.0]
        template2 = [2.0, 4.0, 6.0]  # 2x template1
        A_tSZ = 1.0
        ν1, ν2 = 143.0, 217.0
        ν0 = 143.0
        α_tSZ = 0.0  # No ℓ dependence for simplicity
        ℓ_pivot = 3000
        ℓs = [3000, 3000, 3000]  # At pivot

        D1 = CMBForegrounds.tsz_cross_power(template1, A_tSZ, ν1, ν2, ν0, α_tSZ, ℓ_pivot, ℓs)
        D2 = CMBForegrounds.tsz_cross_power(template2, A_tSZ, ν1, ν2, ν0, α_tSZ, ℓ_pivot, ℓs)

        # Should scale exactly with template
        @test all(D2 .≈ 2.0 .* D1)
    end

    @testset "Amplitude Scaling" begin
        # Test that output scales linearly with amplitude
        template = [10.0, 20.0, 30.0]
        ν1, ν2 = 143.0, 217.0
        ν0 = 143.0
        α_tSZ = 0.0
        ℓ_pivot = 3000
        ℓs = [3000, 3000, 3000]

        A1, A2 = 1.0, 2.5
        D1 = CMBForegrounds.tsz_cross_power(template, A1, ν1, ν2, ν0, α_tSZ, ℓ_pivot, ℓs)
        D2 = CMBForegrounds.tsz_cross_power(template, A2, ν1, ν2, ν0, α_tSZ, ℓ_pivot, ℓs)

        # Should scale exactly with amplitude
        @test all(D2 .≈ (A2 / A1) .* D1)
    end

    @testset "Power Law Scaling" begin
        # Test multipole power law scaling
        template = [1.0, 1.0, 1.0, 1.0]  # Flat template
        A_tSZ = 1.0
        ν1, ν2 = 143.0, 143.0  # Same frequency (auto-spectrum)
        ν0 = 143.0
        ℓ_pivot = 3000
        ℓs = [1500, 3000, 6000, 12000]  # Powers of 2 from pivot

        # Test different power law indices
        α_flat = 0.0
        α_moderate = 0.5
        α_steep = 1.0

        D_flat = CMBForegrounds.tsz_cross_power(template, A_tSZ, ν1, ν2, ν0, α_flat, ℓ_pivot, ℓs)
        D_moderate = CMBForegrounds.tsz_cross_power(template, A_tSZ, ν1, ν2, ν0, α_moderate, ℓ_pivot, ℓs)
        D_steep = CMBForegrounds.tsz_cross_power(template, A_tSZ, ν1, ν2, ν0, α_steep, ℓ_pivot, ℓs)

        # At pivot (ℓ=3000), all should be equal
        @test D_flat[2] ≈ D_moderate[2] ≈ D_steep[2]

        # For flat spectrum (α=0), all values should be equal
        @test all(D_flat[1] .≈ D_flat)

        # For positive α, power should increase with ℓ
        @test D_moderate[1] < D_moderate[2] < D_moderate[3] < D_moderate[4]
        @test D_steep[1] < D_steep[2] < D_steep[3] < D_steep[4]

        # Steeper α should give more extreme scaling
        @test (D_steep[4] / D_steep[1]) > (D_moderate[4] / D_moderate[1])
    end

    @testset "Auto-Spectrum vs Cross-Spectrum" begin
        # Test tSZ auto-spectrum (same frequency) vs cross-spectrum
        template = [5.0, 10.0, 15.0]
        A_tSZ = 1.0
        ν = 143.0
        ν0 = 143.0
        α_tSZ = 0.0
        ℓ_pivot = 3000
        ℓs = [3000, 3000, 3000]

        # Auto-spectrum: same frequency
        D_auto = CMBForegrounds.tsz_cross_power(template, A_tSZ, ν, ν, ν0, α_tSZ, ℓ_pivot, ℓs)

        # Cross-spectrum: different frequencies
        D_cross = CMBForegrounds.tsz_cross_power(template, A_tSZ, 143.0, 217.0, ν0, α_tSZ, ℓ_pivot, ℓs)

        # Auto should use g(ν)^2, cross should use g(ν1)*g(ν2)
        g_143 = CMBForegrounds.tsz_g_ratio(143.0, ν0, CMBForegrounds.T_CMB)
        g_217 = CMBForegrounds.tsz_g_ratio(217.0, ν0, CMBForegrounds.T_CMB)

        # Check auto-spectrum scaling
        expected_auto = template .* A_tSZ .* g_143^2
        @test all(D_auto .≈ expected_auto)

        # Check cross-spectrum scaling
        expected_cross = template .* A_tSZ .* g_143 .* g_217
        @test all(D_cross .≈ expected_cross)
    end

    @testset "Frequency Dependence" begin
        # Test spectral behavior with different frequency combinations
        template = [1.0, 1.0, 1.0]  # Flat template
        A_tSZ = 1.0
        ν0 = 143.0
        α_tSZ = 0.0
        ℓ_pivot = 3000
        ℓs = [3000, 3000, 3000]

        # Different frequency combinations
        freq_pairs = [(143.0, 143.0), (143.0, 217.0), (217.0, 353.0), (353.0, 353.0)]

        D_results = []
        for (ν1, ν2) in freq_pairs
            D = CMBForegrounds.tsz_cross_power(template, A_tSZ, ν1, ν2, ν0, α_tSZ, ℓ_pivot, ℓs)
            push!(D_results, D[1])  # Take first element
            @test all(isfinite.(D))
        end

        # All should be real numbers
        @test all(isa.(D_results, Real))

        # tSZ g-function can be negative, so signs may vary
        # Just check they're all finite
        @test all(isfinite.(D_results))
    end

    @testset "Reference Frequency Effects" begin
        # Test how changing reference frequency affects results
        template = [1.0, 1.0, 1.0]
        A_tSZ = 1.0
        ν1, ν2 = 143.0, 217.0
        α_tSZ = 0.0
        ℓ_pivot = 3000
        ℓs = [3000, 3000, 3000]

        # Different reference frequencies
        ν0_values = [143.0, 217.0, 353.0]

        for ν0 in ν0_values
            D = CMBForegrounds.tsz_cross_power(template, A_tSZ, ν1, ν2, ν0, α_tSZ, ℓ_pivot, ℓs)
            @test all(isfinite.(D))

            # Manual check
            g1 = CMBForegrounds.tsz_g_ratio(ν1, ν0, CMBForegrounds.T_CMB)
            g2 = CMBForegrounds.tsz_g_ratio(ν2, ν0, CMBForegrounds.T_CMB)
            expected = template .* A_tSZ .* g1 .* g2
            @test all(D .≈ expected)
        end
    end

    @testset "Pivot Scale Effects" begin
        # Test how different pivot scales affect the multipole scaling
        template = [1.0, 1.0, 1.0]
        A_tSZ = 1.0
        ν1, ν2 = 143.0, 217.0
        ν0 = 143.0
        α_tSZ = 1.0
        ℓs = [1000, 3000, 6000]

        # Different pivot scales
        ℓ_pivot1 = 2000
        ℓ_pivot2 = 5000

        D1 = CMBForegrounds.tsz_cross_power(template, A_tSZ, ν1, ν2, ν0, α_tSZ, ℓ_pivot1, ℓs)
        D2 = CMBForegrounds.tsz_cross_power(template, A_tSZ, ν1, ν2, ν0, α_tSZ, ℓ_pivot2, ℓs)

        # Results should be different due to different pivot points
        @test D1 != D2
        @test all(isfinite.(D1)) && all(isfinite.(D2))

        # Check that scaling is consistent with formula
        g1 = CMBForegrounds.tsz_g_ratio(ν1, ν0, CMBForegrounds.T_CMB)
        g2 = CMBForegrounds.tsz_g_ratio(ν2, ν0, CMBForegrounds.T_CMB)
        spectral_factor = A_tSZ * g1 * g2

        for i in 1:length(ℓs)
            scale1 = (ℓs[i] / ℓ_pivot1)^α_tSZ
            scale2 = (ℓs[i] / ℓ_pivot2)^α_tSZ
            expected1 = template[i] * spectral_factor * scale1
            expected2 = template[i] * spectral_factor * scale2
            @test D1[i] ≈ expected1
            @test D2[i] ≈ expected2
        end
    end

    @testset "Template Shape Preservation" begin
        # Test that relative template shape is preserved (with scaling)
        template = [1.0, 4.0, 9.0, 16.0]  # Quadratic shape
        A_tSZ = 1.0
        ν1, ν2 = 143.0, 143.0
        ν0 = 143.0
        α_tSZ = 0.0  # No multipole scaling
        ℓ_pivot = 3000
        ℓs = [3000, 3000, 3000, 3000]  # All at pivot

        D = CMBForegrounds.tsz_cross_power(template, A_tSZ, ν1, ν2, ν0, α_tSZ, ℓ_pivot, ℓs)

        # Should preserve relative ratios of template
        g = CMBForegrounds.tsz_g_ratio(ν1, ν0, CMBForegrounds.T_CMB)
        scale_factor = A_tSZ * g^2

        expected = template .* scale_factor
        @test all(D .≈ expected)

        # Check relative ratios are preserved
        @test D[2] / D[1] ≈ template[2] / template[1]
        @test D[4] / D[3] ≈ template[4] / template[3]
    end

    @testset "Type Stability" begin
        # Test with different input types
        template = [1.0, 2.0, 3.0]
        A_tSZ = 1.0
        ν1, ν2 = 143.0, 217.0
        ν0 = 143.0
        α_tSZ = 0.5
        ℓ_pivot = 3000
        ℓs = [1000, 3000, 5000]

        # Float64 inputs
        D_float = CMBForegrounds.tsz_cross_power(template, A_tSZ, ν1, ν2, ν0, α_tSZ, ℓ_pivot, ℓs)
        @test eltype(D_float) == Float64

        # Mixed types - integers and floats
        D_mixed = CMBForegrounds.tsz_cross_power([1, 2, 3], 1, 143, 217.0, 143, 0.5, 3000, [1000, 3000, 5000])
        @test eltype(D_mixed) == Float64
        @test D_mixed ≈ D_float
    end

    @testset "Vector Length Consistency" begin
        # Test that template and ℓs must have same length
        template1 = [1.0, 2.0, 3.0]
        template2 = [1.0, 2.0]
        A_tSZ = 1.0
        ν1, ν2 = 143.0, 217.0
        ν0 = 143.0
        α_tSZ = 0.5
        ℓ_pivot = 3000
        ℓs1 = [1000, 3000, 5000]
        ℓs2 = [1000, 3000]

        # Matching lengths should work
        D1 = CMBForegrounds.tsz_cross_power(template1, A_tSZ, ν1, ν2, ν0, α_tSZ, ℓ_pivot, ℓs1)
        @test length(D1) == length(template1) == length(ℓs1)

        D2 = CMBForegrounds.tsz_cross_power(template2, A_tSZ, ν1, ν2, ν0, α_tSZ, ℓ_pivot, ℓs2)
        @test length(D2) == length(template2) == length(ℓs2)

        # Check that function uses broadcast correctly
        @test all(isfinite.(D1)) && all(isfinite.(D2))
    end

    @testset "Physical Realism" begin
        # Test with realistic tSZ parameters

        # Realistic template based on Planck tSZ measurements
        template = [50.0, 100.0, 150.0, 120.0, 80.0]  # Typical tSZ shape
        A_tSZ = 1.0  # Normalized amplitude
        ν1, ν2 = 143.0, 217.0  # Planck frequencies
        ν0 = 143.0  # Reference
        α_tSZ = 0.8  # Typical tSZ multipole scaling
        ℓ_pivot = 3000
        ℓs = [500, 1500, 3000, 6000, 9000]

        D = CMBForegrounds.tsz_cross_power(template, A_tSZ, ν1, ν2, ν0, α_tSZ, ℓ_pivot, ℓs)

        # Physical expectations
        @test all(isfinite.(D))
        @test length(D) == length(template) == length(ℓs)

        # For positive α, should increase with ℓ beyond template effects
        # Check individual multipole scaling
        g1 = CMBForegrounds.tsz_g_ratio(ν1, ν0, CMBForegrounds.T_CMB)
        g2 = CMBForegrounds.tsz_g_ratio(ν2, ν0, CMBForegrounds.T_CMB)
        spectral_factor = A_tSZ * g1 * g2

        for i in 1:length(ℓs)
            scale_factor = (ℓs[i] / ℓ_pivot)^α_tSZ
            expected = template[i] * spectral_factor * scale_factor
            @test D[i] ≈ expected
        end
    end

    @testset "Custom T_CMB Parameter" begin
        # Test custom CMB temperature parameter
        template = [1.0, 1.0, 1.0]
        A_tSZ = 1.0
        ν1, ν2 = 143.0, 217.0
        ν0 = 143.0
        α_tSZ = 0.0
        ℓ_pivot = 3000
        ℓs = [3000, 3000, 3000]

        # Default T_CMB
        D_default = CMBForegrounds.tsz_cross_power(template, A_tSZ, ν1, ν2, ν0, α_tSZ, ℓ_pivot, ℓs)

        # Custom T_CMB
        custom_T_CMB = 2.8  # Slightly different from default 2.72548
        D_custom = CMBForegrounds.tsz_cross_power(template, A_tSZ, ν1, ν2, ν0, α_tSZ, ℓ_pivot, ℓs; T_CMB=custom_T_CMB)

        # Should be different due to different CMB temperature
        @test D_default != D_custom
        @test all(isfinite.(D_default)) && all(isfinite.(D_custom))

        # Verify with manual calculation
        g1_custom = CMBForegrounds.tsz_g_ratio(ν1, ν0, custom_T_CMB)
        g2_custom = CMBForegrounds.tsz_g_ratio(ν2, ν0, custom_T_CMB)
        expected_custom = template .* A_tSZ .* g1_custom .* g2_custom
        @test all(D_custom .≈ expected_custom)
    end

    @testset "Edge Cases" begin
        # Test with extreme but valid parameter values
        template = [1e-10, 1e10, 1.0]
        ℓs = [100, 3000, 10000]
        ν0 = 143.0
        ℓ_pivot = 3000

        # Very high amplitude
        D_high_A = CMBForegrounds.tsz_cross_power(template, 1e6, 143.0, 217.0, ν0, 0.5, ℓ_pivot, ℓs)
        @test all(isfinite.(D_high_A))

        # Very low amplitude
        D_low_A = CMBForegrounds.tsz_cross_power(template, 1e-6, 143.0, 217.0, ν0, 0.5, ℓ_pivot, ℓs)
        @test all(isfinite.(D_low_A))

        # Extreme power law indices
        D_neg_α = CMBForegrounds.tsz_cross_power(template, 1.0, 143.0, 217.0, ν0, -2.0, ℓ_pivot, ℓs)
        D_pos_α = CMBForegrounds.tsz_cross_power(template, 1.0, 143.0, 217.0, ν0, 3.0, ℓ_pivot, ℓs)
        @test all(isfinite.(D_neg_α)) && all(isfinite.(D_pos_α))

        # Very different pivot
        D_extreme_pivot = CMBForegrounds.tsz_cross_power(template, 1.0, 143.0, 217.0, ν0, 1.0, 100000, ℓs)
        @test all(isfinite.(D_extreme_pivot))
    end

    @testset "Numerical Precision" begin
        # Test numerical precision with challenging values
        template = [1.0]
        A_tSZ = 1.0
        ν1, ν2 = 143.0, 143.0
        ν0 = 143.0
        α_tSZ = 0.0
        ℓ_pivot = 3000

        # Very close to pivot
        ℓs_close = [2999.999, 3000.0, 3000.001]
        D_close = CMBForegrounds.tsz_cross_power([1.0, 1.0, 1.0], A_tSZ, ν1, ν2, ν0, α_tSZ, ℓ_pivot, ℓs_close)

        # Should all be very close since α=0
        @test all(abs.(D_close .- D_close[2]) .< 1e-10)

        # Test with non-zero α
        α_small = 1e-6
        D_small_α = CMBForegrounds.tsz_cross_power([1.0, 1.0, 1.0], A_tSZ, ν1, ν2, ν0, α_small, ℓ_pivot, ℓs_close)
        @test all(isfinite.(D_small_α))
    end
end
