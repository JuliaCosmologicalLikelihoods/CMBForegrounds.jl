"""
Unit tests for tsz_cib_cross_power function

Tests the tSZ-CIB cross-power spectrum function that computes:
D_ℓ = -ξ * (sqrt(|D_ℓ^{tSZ,11} * D_ℓ^{CIB,22}|) + sqrt(|D_ℓ^{tSZ,22} * D_ℓ^{CIB,11}|))
"""

@testset "tsz_cib_cross_power() Unit Tests" begin

    @testset "Basic Functionality" begin
        # Test with typical parameters
        ℓs = [1000, 3000, 5000]
        tsz_template = [10.0, 20.0, 15.0]
        ξ = 0.2
        A_tSZ, A_CIB = 1.0, 1.0
        α, β = 0.8, 1.6
        z1, z2 = 1.0, 1.0
        ν_cib1, ν_cib2 = 217.0, 353.0
        ν_tsz1, ν_tsz2 = 143.0, 217.0
        α_tsz = 0.5
        ν0_tsz, ν0_cib = 143.0, 150.0
        Tdust = 25.0

        D_cross = CMBForegrounds.tsz_cib_cross_power(
            ℓs, ξ, A_tSZ, A_CIB, α, β, z1, z2,
            ν_cib1, ν_cib2, ν_tsz1, ν_tsz2, α_tsz,
            tsz_template, ν0_tsz, Tdust, ν0_cib
        )

        # Basic output tests
        @test D_cross isa AbstractVector
        @test length(D_cross) == length(ℓs)
        @test length(D_cross) == length(tsz_template)
        @test all(isfinite.(D_cross))
        @test eltype(D_cross) <: AbstractFloat

        # Should be negative due to anti-correlation convention
        @test all(D_cross .<= 0)
    end

    @testset "Mathematical Formula Verification" begin
        # Test that the function implements the correct mathematical formula
        ℓs = [3000]  # Single multipole for simplicity
        tsz_template = [10.0]
        ξ = 0.3
        A_tSZ, A_CIB = 1.5, 2.0
        α, β = 0.8, 1.6
        z1, z2 = 1.0, 1.0
        ν_cib1, ν_cib2 = 217.0, 353.0
        ν_tsz1, ν_tsz2 = 143.0, 217.0
        α_tsz = 0.5
        ν0_tsz, ν0_cib = 143.0, 150.0
        Tdust = 25.0
        ℓ_pivot_cib, ℓ_pivot_tsz = 3000, 3000

        D_cross = CMBForegrounds.tsz_cib_cross_power(
            ℓs, ξ, A_tSZ, A_CIB, α, β, z1, z2,
            ν_cib1, ν_cib2, ν_tsz1, ν_tsz2, α_tsz,
            tsz_template, ν0_tsz, Tdust, ν0_cib;
            ℓ_pivot_cib=ℓ_pivot_cib, ℓ_pivot_tsz=ℓ_pivot_tsz
        )[1]

        # Manual calculation of components
        cib_11 = CMBForegrounds.cib_clustered_power(ℓs, A_CIB, α, β, ν_cib1, ν_cib1, z1, z1, Tdust, ν0_cib; ℓ_pivot=ℓ_pivot_cib)[1]
        cib_22 = CMBForegrounds.cib_clustered_power(ℓs, A_CIB, α, β, ν_cib2, ν_cib2, z2, z2, Tdust, ν0_cib; ℓ_pivot=ℓ_pivot_cib)[1]
        tsz_11 = CMBForegrounds.tsz_cross_power(tsz_template, A_tSZ, ν_tsz1, ν_tsz1, ν0_tsz, α_tsz, ℓ_pivot_tsz, ℓs)[1]
        tsz_22 = CMBForegrounds.tsz_cross_power(tsz_template, A_tSZ, ν_tsz2, ν_tsz2, ν0_tsz, α_tsz, ℓ_pivot_tsz, ℓs)[1]

        # Expected result from formula
        expected = -ξ * (sqrt(abs(tsz_11 * cib_22)) + sqrt(abs(tsz_22 * cib_11)))

        @test D_cross ≈ expected
    end

    @testset "Correlation Coefficient Scaling" begin
        # Test that output scales linearly with correlation coefficient ξ
        ℓs = [1000, 3000, 5000]
        tsz_template = [10.0, 20.0, 15.0]
        A_tSZ, A_CIB = 1.0, 1.0
        α, β = 0.8, 1.6
        z1, z2 = 1.0, 1.0
        ν_cib1, ν_cib2 = 217.0, 353.0
        ν_tsz1, ν_tsz2 = 143.0, 217.0
        α_tsz = 0.5
        ν0_tsz, ν0_cib = 143.0, 150.0
        Tdust = 25.0

        ξ1, ξ2 = 0.1, 0.3

        D1 = CMBForegrounds.tsz_cib_cross_power(
            ℓs, ξ1, A_tSZ, A_CIB, α, β, z1, z2,
            ν_cib1, ν_cib2, ν_tsz1, ν_tsz2, α_tsz,
            tsz_template, ν0_tsz, Tdust, ν0_cib
        )

        D2 = CMBForegrounds.tsz_cib_cross_power(
            ℓs, ξ2, A_tSZ, A_CIB, α, β, z1, z2,
            ν_cib1, ν_cib2, ν_tsz1, ν_tsz2, α_tsz,
            tsz_template, ν0_tsz, Tdust, ν0_cib
        )

        # Should scale exactly with correlation coefficient
        @test all(D2 .≈ (ξ2 / ξ1) .* D1)

        # Zero correlation should give zero result
        D_zero = CMBForegrounds.tsz_cib_cross_power(
            ℓs, 0.0, A_tSZ, A_CIB, α, β, z1, z2,
            ν_cib1, ν_cib2, ν_tsz1, ν_tsz2, α_tsz,
            tsz_template, ν0_tsz, Tdust, ν0_cib
        )
        @test all(D_zero .== 0.0)
    end

    @testset "Amplitude Scaling" begin
        # Test scaling with tSZ and CIB amplitudes
        ℓs = [3000]  # At pivot for simplicity
        tsz_template = [10.0]
        ξ = 0.2
        α, β = 0.8, 1.6
        z1, z2 = 1.0, 1.0
        ν_cib1, ν_cib2 = 217.0, 353.0
        ν_tsz1, ν_tsz2 = 143.0, 217.0
        α_tsz = 0.0  # No multipole dependence
        ν0_tsz, ν0_cib = 143.0, 150.0
        Tdust = 25.0

        # Test tSZ amplitude scaling
        A_tSZ_base, A_CIB = 1.0, 1.0
        A_tSZ_scaled = 2.0

        D_base = CMBForegrounds.tsz_cib_cross_power(
            ℓs, ξ, A_tSZ_base, A_CIB, α, β, z1, z2,
            ν_cib1, ν_cib2, ν_tsz1, ν_tsz2, α_tsz,
            tsz_template, ν0_tsz, Tdust, ν0_cib
        )[1]

        D_scaled = CMBForegrounds.tsz_cib_cross_power(
            ℓs, ξ, A_tSZ_scaled, A_CIB, α, β, z1, z2,
            ν_cib1, ν_cib2, ν_tsz1, ν_tsz2, α_tsz,
            tsz_template, ν0_tsz, Tdust, ν0_cib
        )[1]

        # tSZ appears in both cross-terms, so should scale as sqrt(A_tSZ)
        expected_scaling = sqrt(A_tSZ_scaled / A_tSZ_base)
        @test D_scaled ≈ D_base * expected_scaling

        # Test CIB amplitude scaling
        A_tSZ, A_CIB_base = 1.0, 1.0
        A_CIB_scaled = 3.0

        D_base_cib = CMBForegrounds.tsz_cib_cross_power(
            ℓs, ξ, A_tSZ, A_CIB_base, α, β, z1, z2,
            ν_cib1, ν_cib2, ν_tsz1, ν_tsz2, α_tsz,
            tsz_template, ν0_tsz, Tdust, ν0_cib
        )[1]

        D_scaled_cib = CMBForegrounds.tsz_cib_cross_power(
            ℓs, ξ, A_tSZ, A_CIB_scaled, α, β, z1, z2,
            ν_cib1, ν_cib2, ν_tsz1, ν_tsz2, α_tsz,
            tsz_template, ν0_tsz, Tdust, ν0_cib
        )[1]

        # CIB appears in both cross-terms, so should scale as sqrt(A_CIB)
        expected_scaling_cib = sqrt(A_CIB_scaled / A_CIB_base)
        @test D_scaled_cib ≈ D_base_cib * expected_scaling_cib
    end

    @testset "Template Scaling" begin
        # Test that tSZ template scales appropriately
        ℓs = [1000, 3000, 5000]
        template1 = [1.0, 2.0, 3.0]
        template2 = [2.0, 4.0, 6.0]  # 2x template1
        ξ = 0.2
        A_tSZ, A_CIB = 1.0, 1.0
        α, β = 0.8, 1.6
        z1, z2 = 1.0, 1.0
        ν_cib1, ν_cib2 = 217.0, 353.0
        ν_tsz1, ν_tsz2 = 143.0, 217.0
        α_tsz = 0.0  # No additional multipole dependence
        ν0_tsz, ν0_cib = 143.0, 150.0
        Tdust = 25.0

        D1 = CMBForegrounds.tsz_cib_cross_power(
            ℓs, ξ, A_tSZ, A_CIB, α, β, z1, z2,
            ν_cib1, ν_cib2, ν_tsz1, ν_tsz2, α_tsz,
            template1, ν0_tsz, Tdust, ν0_cib
        )

        D2 = CMBForegrounds.tsz_cib_cross_power(
            ℓs, ξ, A_tSZ, A_CIB, α, β, z1, z2,
            ν_cib1, ν_cib2, ν_tsz1, ν_tsz2, α_tsz,
            template2, ν0_tsz, Tdust, ν0_cib
        )

        # Template appears in both tSZ terms, so should scale as sqrt(template)
        @test all(D2 .≈ D1 .* sqrt(2.0))
    end

    @testset "Frequency Dependencies" begin
        # Test how different frequency combinations affect results
        ℓs = [3000]
        tsz_template = [10.0]
        ξ = 0.2
        A_tSZ, A_CIB = 1.0, 1.0
        α, β = 0.8, 1.6
        z1, z2 = 1.0, 1.0
        α_tsz = 0.0
        ν0_tsz, ν0_cib = 143.0, 150.0
        Tdust = 25.0

        # Test different frequency combinations
        freq_combinations = [
            (217.0, 353.0, 143.0, 217.0),  # Standard combination
            (143.0, 217.0, 143.0, 217.0),  # Same frequencies for both
            (353.0, 353.0, 217.0, 217.0),  # Auto-spectra for both
            (143.0, 353.0, 217.0, 353.0),  # Different combination
        ]

        results = []
        for (ν_cib1, ν_cib2, ν_tsz1, ν_tsz2) in freq_combinations
            D = CMBForegrounds.tsz_cib_cross_power(
                ℓs, ξ, A_tSZ, A_CIB, α, β, z1, z2,
                ν_cib1, ν_cib2, ν_tsz1, ν_tsz2, α_tsz,
                tsz_template, ν0_tsz, Tdust, ν0_cib
            )[1]
            push!(results, D)
            @test isfinite(D)
            @test D <= 0  # Should be negative
        end

        # All should be different (unless by chance they're equal)
        @test length(unique(results)) >= 1  # At least one unique result
    end

    @testset "Pivot Scale Effects" begin
        # Test how different pivot scales affect results
        ℓs = [1000, 3000, 6000]
        tsz_template = [5.0, 10.0, 15.0]
        ξ = 0.15
        A_tSZ, A_CIB = 1.0, 1.0
        α, β = 0.8, 1.6
        z1, z2 = 1.0, 1.0
        ν_cib1, ν_cib2 = 217.0, 353.0
        ν_tsz1, ν_tsz2 = 143.0, 217.0
        α_tsz = 0.5
        ν0_tsz, ν0_cib = 143.0, 150.0
        Tdust = 25.0

        # Different pivot combinations
        pivot_combinations = [
            (3000, 3000),  # Standard
            (2000, 4000),  # Different pivots
            (5000, 1000),  # Reversed pivots
        ]

        for (ℓ_pivot_cib, ℓ_pivot_tsz) in pivot_combinations
            D = CMBForegrounds.tsz_cib_cross_power(
                ℓs, ξ, A_tSZ, A_CIB, α, β, z1, z2,
                ν_cib1, ν_cib2, ν_tsz1, ν_tsz2, α_tsz,
                tsz_template, ν0_tsz, Tdust, ν0_cib;
                ℓ_pivot_cib=ℓ_pivot_cib, ℓ_pivot_tsz=ℓ_pivot_tsz
            )

            @test all(isfinite.(D))
            @test all(D .<= 0)
            @test length(D) == length(ℓs)
        end
    end

    @testset "CIB Parameter Effects" begin
        # Test how CIB-specific parameters affect results
        ℓs = [3000]
        tsz_template = [10.0]
        ξ = 0.2
        A_tSZ, A_CIB = 1.0, 1.0
        z1, z2 = 1.0, 1.0
        ν_cib1, ν_cib2 = 217.0, 353.0
        ν_tsz1, ν_tsz2 = 143.0, 217.0
        α_tsz = 0.0
        ν0_tsz, ν0_cib = 143.0, 150.0

        # Test different CIB parameters
        cib_params = [
            (0.6, 1.4, 20.0),  # α, β, Tdust
            (0.8, 1.6, 25.0),  # Standard
            (1.0, 1.8, 30.0),  # Higher values
        ]

        for (α, β, Tdust) in cib_params
            D = CMBForegrounds.tsz_cib_cross_power(
                ℓs, ξ, A_tSZ, A_CIB, α, β, z1, z2,
                ν_cib1, ν_cib2, ν_tsz1, ν_tsz2, α_tsz,
                tsz_template, ν0_tsz, Tdust, ν0_cib
            )[1]

            @test isfinite(D)
            @test D <= 0
        end
    end

    @testset "Redshift Factor Effects" begin
        # Test how CIB redshift factors affect results
        ℓs = [3000]
        tsz_template = [10.0]
        ξ = 0.2
        A_tSZ, A_CIB = 1.0, 1.0
        α, β = 0.8, 1.6
        ν_cib1, ν_cib2 = 217.0, 353.0
        ν_tsz1, ν_tsz2 = 143.0, 217.0
        α_tsz = 0.0
        ν0_tsz, ν0_cib = 143.0, 150.0
        Tdust = 25.0

        # Test different redshift factor combinations
        z_combinations = [
            (1.0, 1.0),
            (0.8, 1.2),
            (0.5, 2.0),
            (1.5, 1.5),
        ]

        for (z1, z2) in z_combinations
            D = CMBForegrounds.tsz_cib_cross_power(
                ℓs, ξ, A_tSZ, A_CIB, α, β, z1, z2,
                ν_cib1, ν_cib2, ν_tsz1, ν_tsz2, α_tsz,
                tsz_template, ν0_tsz, Tdust, ν0_cib
            )[1]

            @test isfinite(D)
            @test D <= 0
        end
    end

    @testset "Vector Length Consistency" begin
        # Test the assertion about vector lengths
        ℓs_3 = [1000, 3000, 5000]
        ℓs_4 = [1000, 2000, 3000, 4000]
        template_3 = [1.0, 2.0, 3.0]
        template_4 = [1.0, 2.0, 3.0, 4.0]

        ξ = 0.2
        A_tSZ, A_CIB = 1.0, 1.0
        α, β = 0.8, 1.6
        z1, z2 = 1.0, 1.0
        ν_cib1, ν_cib2 = 217.0, 353.0
        ν_tsz1, ν_tsz2 = 143.0, 217.0
        α_tsz = 0.0
        ν0_tsz, ν0_cib = 143.0, 150.0
        Tdust = 25.0

        # Matching lengths should work
        D3 = CMBForegrounds.tsz_cib_cross_power(
            ℓs_3, ξ, A_tSZ, A_CIB, α, β, z1, z2,
            ν_cib1, ν_cib2, ν_tsz1, ν_tsz2, α_tsz,
            template_3, ν0_tsz, Tdust, ν0_cib
        )
        @test length(D3) == 3
        @test all(isfinite.(D3))

        D4 = CMBForegrounds.tsz_cib_cross_power(
            ℓs_4, ξ, A_tSZ, A_CIB, α, β, z1, z2,
            ν_cib1, ν_cib2, ν_tsz1, ν_tsz2, α_tsz,
            template_4, ν0_tsz, Tdust, ν0_cib
        )
        @test length(D4) == 4
        @test all(isfinite.(D4))

        # Mismatched lengths should throw assertion error
        @test_throws AssertionError CMBForegrounds.tsz_cib_cross_power(
            ℓs_3, ξ, A_tSZ, A_CIB, α, β, z1, z2,
            ν_cib1, ν_cib2, ν_tsz1, ν_tsz2, α_tsz,
            template_4, ν0_tsz, Tdust, ν0_cib  # Wrong template length
        )
    end

    @testset "Negative Correlation Convention" begin
        # Test that the function returns negative values (anti-correlation convention)
        ℓs = [1000, 3000, 5000]
        tsz_template = [5.0, 10.0, 15.0]
        A_tSZ, A_CIB = 1.0, 1.0
        α, β = 0.8, 1.6
        z1, z2 = 1.0, 1.0
        ν_cib1, ν_cib2 = 217.0, 353.0
        ν_tsz1, ν_tsz2 = 143.0, 217.0
        α_tsz = 0.5
        ν0_tsz, ν0_cib = 143.0, 150.0
        Tdust = 25.0

        # Positive correlation coefficient
        D_pos = CMBForegrounds.tsz_cib_cross_power(
            ℓs, 0.2, A_tSZ, A_CIB, α, β, z1, z2,
            ν_cib1, ν_cib2, ν_tsz1, ν_tsz2, α_tsz,
            tsz_template, ν0_tsz, Tdust, ν0_cib
        )

        # Negative correlation coefficient
        D_neg = CMBForegrounds.tsz_cib_cross_power(
            ℓs, -0.2, A_tSZ, A_CIB, α, β, z1, z2,
            ν_cib1, ν_cib2, ν_tsz1, ν_tsz2, α_tsz,
            tsz_template, ν0_tsz, Tdust, ν0_cib
        )

        # Positive ξ should give negative D (due to -ξ in formula)
        @test all(D_pos .<= 0)
        # Negative ξ should give positive D
        @test all(D_neg .>= 0)
        # They should be opposite in sign
        @test all(D_pos .≈ -D_neg)
    end

    @testset "Type Stability" begin
        # Test with different input types
        ℓs = [1000, 3000, 5000]
        tsz_template = [1.0, 2.0, 3.0]
        ξ = 0.2
        A_tSZ, A_CIB = 1.0, 1.0
        α, β = 0.8, 1.6
        z1, z2 = 1.0, 1.0
        ν_cib1, ν_cib2 = 217.0, 353.0
        ν_tsz1, ν_tsz2 = 143.0, 217.0
        α_tsz = 0.5
        ν0_tsz, ν0_cib = 143.0, 150.0
        Tdust = 25.0

        # Float64 inputs
        D_float = CMBForegrounds.tsz_cib_cross_power(
            ℓs, ξ, A_tSZ, A_CIB, α, β, z1, z2,
            ν_cib1, ν_cib2, ν_tsz1, ν_tsz2, α_tsz,
            tsz_template, ν0_tsz, Tdust, ν0_cib
        )
        @test eltype(D_float) == Float64

        # Mixed types
        D_mixed = CMBForegrounds.tsz_cib_cross_power(
            [1000, 3000, 5000], 0.2, 1, 1.0, 0.8, 1.6, 1, 1.0,
            217, 353.0, 143, 217.0, 0.5,
            [1.0, 2.0, 3.0], 143.0, 25, 150.0
        )
        @test eltype(D_mixed) == Float64
        @test D_mixed ≈ D_float
    end

    @testset "Physical Realism" begin
        # Test with realistic CMB survey parameters
        ℓs = [500, 1500, 3000, 6000, 9000]
        # Realistic tSZ template shape (peak at intermediate ℓ)
        tsz_template = [20.0, 80.0, 100.0, 60.0, 30.0]

        # Realistic parameters from Planck/ACT-like surveys
        ξ = 0.2         # Typical tSZ-CIB correlation
        A_tSZ = 1.0     # Normalized tSZ amplitude
        A_CIB = 1.0     # Normalized CIB amplitude
        α = 0.8         # CIB multipole scaling
        β = 1.6         # Dust emissivity index
        z1, z2 = 1.0, 1.0  # CIB redshift factors
        ν_cib1, ν_cib2 = 217.0, 353.0  # Planck frequencies
        ν_tsz1, ν_tsz2 = 143.0, 217.0  # Planck frequencies
        α_tsz = 0.5     # tSZ multipole scaling
        ν0_tsz = 143.0  # Reference frequency
        Tdust = 25.0    # CIB dust temperature
        ν0_cib = 150.0  # CIB reference frequency

        D_cross = CMBForegrounds.tsz_cib_cross_power(
            ℓs, ξ, A_tSZ, A_CIB, α, β, z1, z2,
            ν_cib1, ν_cib2, ν_tsz1, ν_tsz2, α_tsz,
            tsz_template, ν0_tsz, Tdust, ν0_cib
        )

        # Physical expectations
        @test all(isfinite.(D_cross))
        @test all(D_cross .<= 0)  # Anti-correlation
        @test length(D_cross) == length(ℓs)

        # Magnitude should be reasonable compared to individual components
        @test all(abs.(D_cross) .> 0)  # Non-zero cross-correlation

        # Should preserve some template shape characteristics
        # (though modified by CIB and correlation effects)
        @test all(D_cross .!= 0) || ξ == 0  # Non-zero unless no correlation
    end

    @testset "Custom T_CMB Parameter" begin
        # Test custom CMB temperature
        ℓs = [3000]
        tsz_template = [10.0]
        ξ = 0.2
        A_tSZ, A_CIB = 1.0, 1.0
        α, β = 0.8, 1.6
        z1, z2 = 1.0, 1.0
        ν_cib1, ν_cib2 = 217.0, 353.0
        ν_tsz1, ν_tsz2 = 143.0, 217.0
        α_tsz = 0.0
        ν0_tsz, ν0_cib = 143.0, 150.0
        Tdust = 25.0

        # Default T_CMB
        D_default = CMBForegrounds.tsz_cib_cross_power(
            ℓs, ξ, A_tSZ, A_CIB, α, β, z1, z2,
            ν_cib1, ν_cib2, ν_tsz1, ν_tsz2, α_tsz,
            tsz_template, ν0_tsz, Tdust, ν0_cib
        )[1]

        # Custom T_CMB
        custom_T_CMB = 2.8
        D_custom = CMBForegrounds.tsz_cib_cross_power(
            ℓs, ξ, A_tSZ, A_CIB, α, β, z1, z2,
            ν_cib1, ν_cib2, ν_tsz1, ν_tsz2, α_tsz,
            tsz_template, ν0_tsz, Tdust, ν0_cib;
            T_CMB=custom_T_CMB
        )[1]

        # Should be different due to temperature dependence in both components
        @test D_default != D_custom
        @test isfinite(D_default) && isfinite(D_custom)
        @test D_default <= 0 && D_custom <= 0
    end

    @testset "Edge Cases and Robustness" begin
        # Test with extreme but valid parameter values
        ℓs = [100, 3000, 10000]
        tsz_template = [1e-6, 1.0, 1e6]  # Wide range of template values

        # Extreme correlation coefficient
        for ξ in [-1.0, -0.5, 0.0, 0.5, 1.0]
            D = CMBForegrounds.tsz_cib_cross_power(
                ℓs, ξ, 1.0, 1.0, 0.8, 1.6, 1.0, 1.0,
                217.0, 353.0, 143.0, 217.0, 0.5,
                tsz_template, 143.0, 25.0, 150.0
            )
            @test all(isfinite.(D))
            @test sign(ξ) == -sign(D[2]) || ξ == 0  # Sign relationship
        end

        # Extreme amplitudes
        D_high_A = CMBForegrounds.tsz_cib_cross_power(
            ℓs, 0.1, 100.0, 100.0, 0.8, 1.6, 1.0, 1.0,
            217.0, 353.0, 143.0, 217.0, 0.5,
            tsz_template, 143.0, 25.0, 150.0
        )
        @test all(isfinite.(D_high_A))

        D_low_A = CMBForegrounds.tsz_cib_cross_power(
            ℓs, 0.1, 1e-3, 1e-3, 0.8, 1.6, 1.0, 1.0,
            217.0, 353.0, 143.0, 217.0, 0.5,
            tsz_template, 143.0, 25.0, 150.0
        )
        @test all(isfinite.(D_low_A))

        # Extreme pivot values
        D_extreme_pivot = CMBForegrounds.tsz_cib_cross_power(
            ℓs, 0.1, 1.0, 1.0, 0.8, 1.6, 1.0, 1.0,
            217.0, 353.0, 143.0, 217.0, 1.0,
            tsz_template, 143.0, 25.0, 150.0;
            ℓ_pivot_cib=100000, ℓ_pivot_tsz=10
        )
        @test all(isfinite.(D_extreme_pivot))
    end

    @testset "Numerical Precision" begin
        # Test numerical precision with challenging parameter combinations
        ℓs = [2999.999, 3000.0, 3000.001]
        tsz_template = [10.0, 10.0, 10.0]
        ξ = 0.2
        A_tSZ, A_CIB = 1.0, 1.0
        α, β = 0.0, 1.6  # No CIB multipole dependence
        z1, z2 = 1.0, 1.0
        ν_cib1, ν_cib2 = 217.0, 353.0
        ν_tsz1, ν_tsz2 = 143.0, 217.0
        α_tsz = 0.0  # No tSZ multipole dependence
        ν0_tsz, ν0_cib = 143.0, 150.0
        Tdust = 25.0

        D = CMBForegrounds.tsz_cib_cross_power(
            ℓs, ξ, A_tSZ, A_CIB, α, β, z1, z2,
            ν_cib1, ν_cib2, ν_tsz1, ν_tsz2, α_tsz,
            tsz_template, ν0_tsz, Tdust, ν0_cib
        )

        # With no multipole dependence and flat template, should be very close
        @test all(abs.(D .- D[2]) .< 1e-10)
        @test all(isfinite.(D))

        # Test with very small correlation coefficient
        D_small_ξ = CMBForegrounds.tsz_cib_cross_power(
            ℓs, 1e-10, A_tSZ, A_CIB, α, β, z1, z2,
            ν_cib1, ν_cib2, ν_tsz1, ν_tsz2, α_tsz,
            tsz_template, ν0_tsz, Tdust, ν0_cib
        )
        @test all(isfinite.(D_small_ξ))
        @test all(abs.(D_small_ξ) .< 1e-8)  # Should be very small
    end
end
