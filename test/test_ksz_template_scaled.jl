"""
Unit tests for ksz_template_scaled function

Tests the kinematic Sunyaev-Zel'dovich (kSZ) template scaling function that computes:
D_ℓ = template * AkSZ
"""

@testset "ksz_template_scaled() Unit Tests" begin

    @testset "Basic Functionality" begin
        # Test with typical kSZ template
        template = [10.0, 20.0, 50.0, 80.0, 60.0]
        AkSZ = 1.5

        D_scaled = CMBForegrounds.ksz_template_scaled(template, AkSZ)

        # Basic output tests
        @test D_scaled isa AbstractVector
        @test length(D_scaled) == length(template)
        @test all(isfinite.(D_scaled))
        @test eltype(D_scaled) <: AbstractFloat

        # Should be exactly template * amplitude
        expected = template .* AkSZ
        @test all(D_scaled .≈ expected)
    end

    @testset "Linear Scaling Properties" begin
        # Test that scaling is exactly linear
        template = [1.0, 2.0, 3.0, 4.0]

        A1, A2 = 1.0, 2.5
        D1 = CMBForegrounds.ksz_template_scaled(template, A1)
        D2 = CMBForegrounds.ksz_template_scaled(template, A2)

        # Should scale exactly linearly
        @test all(D2 .≈ (A2 / A1) .* D1)
        @test all(D2 .≈ template .* A2)
        @test all(D1 .≈ template .* A1)
    end

    @testset "Mathematical Properties" begin
        # Test mathematical properties of linear scaling
        template = [5.0, 10.0, 15.0]

        # Zero amplitude gives zero result
        D_zero = CMBForegrounds.ksz_template_scaled(template, 0.0)
        @test all(D_zero .== 0.0)

        # Unit amplitude preserves template
        D_unit = CMBForegrounds.ksz_template_scaled(template, 1.0)
        @test all(D_unit .≈ template)

        # Negative amplitude reverses sign
        A_pos, A_neg = 2.0, -2.0
        D_pos = CMBForegrounds.ksz_template_scaled(template, A_pos)
        D_neg = CMBForegrounds.ksz_template_scaled(template, A_neg)
        @test all(D_pos .≈ -D_neg)

        # Fractional amplitude scales down
        A_frac = 0.5
        D_frac = CMBForegrounds.ksz_template_scaled(template, A_frac)
        @test all(D_frac .≈ template .* A_frac)
        @test all(abs.(D_frac) .< abs.(template))  # Scaled down
    end

    @testset "Type Stability and Promotion" begin
        # Test with different input types
        template = [1.0, 2.0, 3.0]

        # Float64 inputs
        D_float = CMBForegrounds.ksz_template_scaled(template, 1.5)
        @test eltype(D_float) == Float64

        # Integer amplitude
        D_int = CMBForegrounds.ksz_template_scaled(template, 2)
        @test eltype(D_int) == Float64
        @test all(D_int .≈ template .* 2.0)

        # Mixed types - integer template, float amplitude
        template_int = [1, 2, 3]
        D_mixed = CMBForegrounds.ksz_template_scaled(template_int, 1.5)
        @test eltype(D_mixed) == Float64
        @test all(D_mixed .≈ [1.5, 3.0, 4.5])

        # Both integers
        D_both_int = CMBForegrounds.ksz_template_scaled([2, 4, 6], 3)
        @test eltype(D_both_int) == Int
        @test all(D_both_int .== [6, 12, 18])
    end

    @testset "Vector Operations" begin
        # Test with different vector lengths
        templates = [
            [1.0],                    # Single element
            [1.0, 2.0],              # Two elements
            [1.0, 2.0, 3.0, 4.0, 5.0], # Five elements
            collect(1.0:10.0)         # Ten elements
        ]

        AkSZ = 2.0

        for template in templates
            D = CMBForegrounds.ksz_template_scaled(template, AkSZ)
            @test length(D) == length(template)
            @test all(D .≈ template .* AkSZ)
            @test eltype(D) == Float64
        end

        # Test that each element scales independently
        template = [1.0, 10.0, 100.0]
        AkSZ = 3.0
        D = CMBForegrounds.ksz_template_scaled(template, AkSZ)
        @test D[1] ≈ 3.0
        @test D[2] ≈ 30.0
        @test D[3] ≈ 300.0
    end

    @testset "Edge Cases" begin
        # Test with extreme amplitude values
        template = [1.0, 2.0, 3.0]

        # Very large amplitude
        A_large = 1e10
        D_large = CMBForegrounds.ksz_template_scaled(template, A_large)
        @test all(isfinite.(D_large))
        @test all(D_large .≈ template .* A_large)

        # Very small amplitude
        A_small = 1e-10
        D_small = CMBForegrounds.ksz_template_scaled(template, A_small)
        @test all(isfinite.(D_small))
        @test all(D_small .≈ template .* A_small)
        @test all(abs.(D_small) .< 1e-9)

        # Negative amplitude
        A_negative = -5.0
        D_negative = CMBForegrounds.ksz_template_scaled(template, A_negative)
        @test all(isfinite.(D_negative))
        @test all(D_negative .< 0)  # Should be negative
        @test all(D_negative .≈ template .* A_negative)
    end

    @testset "Special Values" begin
        # Test with special float values in template
        AkSZ = 2.0

        # Template with zeros
        template_zeros = [0.0, 1.0, 0.0, 2.0]
        D_zeros = CMBForegrounds.ksz_template_scaled(template_zeros, AkSZ)
        @test D_zeros[1] == 0.0
        @test D_zeros[3] == 0.0
        @test D_zeros[2] ≈ 2.0
        @test D_zeros[4] ≈ 4.0

        # Template with negative values
        template_negative = [-1.0, 2.0, -3.0]
        D_negative = CMBForegrounds.ksz_template_scaled(template_negative, AkSZ)
        @test D_negative[1] ≈ -2.0
        @test D_negative[2] ≈ 4.0
        @test D_negative[3] ≈ -6.0

        # Zero amplitude with any template
        template_any = [100.0, -50.0, 25.0]
        D_zero_amp = CMBForegrounds.ksz_template_scaled(template_any, 0.0)
        @test all(D_zero_amp .== 0.0)
    end

    @testset "Physical Realism" begin
        # Test with realistic kSZ parameters

        # Realistic kSZ template based on theoretical predictions
        # kSZ typically peaks at intermediate ℓ and falls off at high ℓ
        ℓs = [500, 1000, 2000, 3000, 5000, 8000]
        # Approximate kSZ shape: rises then falls
        template_realistic = [50.0, 100.0, 150.0, 120.0, 80.0, 40.0]

        # Realistic kSZ amplitudes from surveys
        AkSZ_values = [0.5, 1.0, 2.0, 5.0]  # Typical range for kSZ amplitude

        for AkSZ in AkSZ_values
            D = CMBForegrounds.ksz_template_scaled(template_realistic, AkSZ)

            # Physical expectations
            @test all(isfinite.(D))
            @test length(D) == length(template_realistic)
            @test all(D .≈ template_realistic .* AkSZ)

            # Should preserve template shape relative to amplitude
            if AkSZ > 0
                # Relative ratios should be preserved
                for i in 2:length(D)
                    expected_ratio = template_realistic[i] / template_realistic[1]
                    actual_ratio = D[i] / D[1]
                    @test actual_ratio ≈ expected_ratio
                end
            end
        end
    end

    @testset "Broadcast Behavior" begin
        # Test that the function uses broadcasting correctly
        template = [1.0, 2.0, 3.0]
        AkSZ = 4.0

        D = CMBForegrounds.ksz_template_scaled(template, AkSZ)

        # Should be equivalent to manual broadcast
        manual_broadcast = @. template * AkSZ
        @test all(D .≈ manual_broadcast)

        # Test with different shaped inputs
        template_row = [1.0 2.0 3.0]  # Row vector (1x3 matrix)
        # Note: This might fail if function requires AbstractVector, but let's see
        # D_row = CMBForegrounds.ksz_template_scaled(template_row, AkSZ)
        # For now, skip this test since function signature requires AbstractVector
    end

    @testset "Numerical Precision" begin
        # Test numerical precision with challenging values

        # Very small template values
        template_small = [1e-15, 2e-15, 3e-15]
        AkSZ = 1e6
        D_small = CMBForegrounds.ksz_template_scaled(template_small, AkSZ)
        @test all(isfinite.(D_small))
        @test all(D_small .≈ template_small .* AkSZ)

        # Very precise calculations
        template_precise = [π, ℯ, sqrt(2)]
        AkSZ_precise = sqrt(3)
        D_precise = CMBForegrounds.ksz_template_scaled(template_precise, AkSZ_precise)

        expected_precise = [π * sqrt(3), ℯ * sqrt(3), sqrt(2) * sqrt(3)]
        @test all(D_precise .≈ expected_precise)

        # Test precision is maintained
        for i in 1:3
            @test D_precise[i] / AkSZ_precise ≈ template_precise[i]
        end
    end

    @testset "Memory and Performance" begin
        # Test that function doesn't unnecessarily allocate or copy

        # Large template
        n = 1000
        template_large = randn(n)
        AkSZ = 2.5

        D_large = CMBForegrounds.ksz_template_scaled(template_large, AkSZ)

        # Basic correctness
        @test length(D_large) == n
        @test all(isfinite.(D_large))
        @test all(D_large .≈ template_large .* AkSZ)

        # Should be efficient - no special checks needed, just verify it works
        @test eltype(D_large) == Float64
    end

    @testset "Consistency with Manual Calculation" begin
        # Test against manual element-by-element calculation
        template = [1.5, -2.3, 0.0, 4.7, -1.1]
        AkSZ = 3.2

        D = CMBForegrounds.ksz_template_scaled(template, AkSZ)

        # Manual calculation
        manual_result = similar(template)
        for i in 1:length(template)
            manual_result[i] = template[i] * AkSZ
        end

        # Should be identical
        @test all(D .≈ manual_result)
        @test all(D .== manual_result)  # Exact equality for this simple operation

        # Test specific values
        @test D[1] ≈ 1.5 * 3.2
        @test D[2] ≈ -2.3 * 3.2
        @test D[3] ≈ 0.0 * 3.2
        @test D[4] ≈ 4.7 * 3.2
        @test D[5] ≈ -1.1 * 3.2
    end

    @testset "Amplitude Sign Effects" begin
        # Test how positive and negative amplitudes affect results
        template = [10.0, -5.0, 20.0, -15.0]

        A_pos = 2.0
        A_neg = -2.0

        D_pos = CMBForegrounds.ksz_template_scaled(template, A_pos)
        D_neg = CMBForegrounds.ksz_template_scaled(template, A_neg)

        # Negative amplitude should flip all signs
        @test all(D_pos .≈ -D_neg)

        # Check individual elements
        @test D_pos[1] > 0 && D_neg[1] < 0  # 10.0 -> positive/negative
        @test D_pos[2] < 0 && D_neg[2] > 0  # -5.0 -> negative/positive
        @test D_pos[3] > 0 && D_neg[3] < 0  # 20.0 -> positive/negative
        @test D_pos[4] < 0 && D_neg[4] > 0  # -15.0 -> negative/positive

        # Magnitudes should be the same
        @test all(abs.(D_pos) .≈ abs.(D_neg))
    end

    @testset "Empty Vector Edge Case" begin
        # Test with empty template vector
        template_empty = Float64[]
        AkSZ = 5.0

        D_empty = CMBForegrounds.ksz_template_scaled(template_empty, AkSZ)
        @test length(D_empty) == 0
        @test D_empty isa Vector{Float64}
    end
end
