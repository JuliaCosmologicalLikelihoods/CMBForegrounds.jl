using Test
using CMBForegrounds

@testset "New Hillipop Foreground Functions" begin
    ℓs = collect(2:2500)

    # ---------------------------------------------------------
    # dust_model_template_power
    # ---------------------------------------------------------
    @testset "dust_model_template_power" begin
        # Create a flat unit template
        tmpl = ones(Float64, 2501)
        # At the reference frequencies/params the output should be A1*A2 * s1*s2
        # with s1,s2 evaluated at same freq → s² = cib_mbb_sed_weight(β,T,ν0,ν0)² = 1
        A1, A2 = 1.0, 1.0
        β1 = β2 = 1.5
        ν0 = 370.5   # effective 353 GHz dust ref
        ν1 = 370.5   # same frequency → ratio = 1
        ν2 = 370.5

        result = dust_model_template_power(ℓs, tmpl[3:end], A1, A2, β1, β2, ν1, ν2, ν0, 19.6)
        @test length(result) == length(ℓs)
        # At ν=ν0=ν_ref, MBB SED ratio = 1 → result = A1*A2 * template = 1
        @test all(isapprox.(result, 1.0; atol=1e-10))

        # Amplitude scaling
        result2 = dust_model_template_power(ℓs, tmpl[3:end], 2.0, 3.0, β1, β2, ν1, ν2, ν0, 19.6)
        @test all(isapprox.(result2, 6.0; atol=1e-10))

        # Non-trivial frequency ratio: result should be positive
        res3 = dust_model_template_power(ℓs, tmpl[3:end], 1.0, 1.0, 1.5, 1.5,
                                          228.1, 228.1, 370.5, 19.6)
        @test all(res3 .> 0.0)
    end

    # ---------------------------------------------------------
    # radio_ps_power
    # ---------------------------------------------------------
    @testset "radio_ps_power" begin
        # At ν=ν0, radioRatio = 1 → result = A * (ℓ(ℓ+1))/(ℓ_pivot(ℓ_pivot+1))
        result = radio_ps_power(ℓs, 1.0, -0.7, 143.0, 143.0, 143.0; ℓ_pivot=3000)
        expected = @. ℓs * (ℓs + 1) / (3000.0 * 3001.0)
        @test length(result) == length(ℓs)
        @test all(isapprox.(result, expected; rtol=1e-8))

        # Amplitude scaling
        result2 = radio_ps_power(ℓs, 2.5, -0.7, 143.0, 143.0, 143.0; ℓ_pivot=3000)
        @test all(isapprox.(result2 ./ result, 2.5; rtol=1e-10))

        # Different frequencies give a positive result
        result3 = radio_ps_power(ℓs, 1.0, -0.7, 100.4, 218.6, 143.0; ℓ_pivot=3000)
        @test all(result3 .> 0.0)

        # Power law: increasing β_radio increases 217/100 ratio
        r_low  = radio_ps_power([3000], 1.0, -0.7, 217.0, 217.0, 143.0)[1]
        r_high = radio_ps_power([3000], 1.0, -0.5, 217.0, 217.0, 143.0)[1]
        # Steeper index (less negative) increases high-freq relative to ref
        @test r_high > r_low
    end

    # ---------------------------------------------------------
    # dusty_ps_power
    # ---------------------------------------------------------
    @testset "dusty_ps_power" begin
        # At ν=ν0, cib SED ratio = 1 → result = A * (ℓ(ℓ+1))/(ℓ_pivot(ℓ_pivot+1))
        result = dusty_ps_power(ℓs, 1.0, 1.75, 143.0, 143.0, 143.0, 25.0; ℓ_pivot=3000)
        expected = @. ℓs * (ℓs + 1) / (3000.0 * 3001.0)
        @test length(result) == length(ℓs)
        @test all(isapprox.(result, expected; rtol=1e-8))

        # Amplitude scaling
        result2 = dusty_ps_power(ℓs, 3.0, 1.75, 143.0, 143.0, 143.0, 25.0; ℓ_pivot=3000)
        @test all(isapprox.(result2 ./ result, 3.0; rtol=1e-10))

        # Different frequencies give a positive result
        result3 = dusty_ps_power(ℓs, 1.0, 1.75, 228.1, 147.5, 143.0, 25.0; ℓ_pivot=3000)
        @test all(result3 .> 0.0)
    end

    # ---------------------------------------------------------
    # sub_pixel_power
    # ---------------------------------------------------------
    @testset "sub_pixel_power" begin
        # With identical FWHMs, check shape ∝ ℓ(ℓ+1) / B_ℓ²
        fwhm = 7.30   # 143 GHz
        result = sub_pixel_power(ℓs, 1.0, fwhm, fwhm; ℓ_pivot=3000)
        @test length(result) == length(ℓs)
        @test all(result .> 0.0)

        # At ℓ→0, beam→1 → result → 0 (numerator → 0 faster)
        result_small_ℓ = sub_pixel_power([2], 1.0, fwhm, fwhm; ℓ_pivot=3000)
        @test result_small_ℓ[1] ≥ 0.0

        # Amplitude scaling
        result2 = sub_pixel_power(ℓs, 2.0, fwhm, fwhm; ℓ_pivot=3000)
        @test all(isapprox.(result2 ./ result, 2.0; rtol=1e-10))

        # Wider beam gives larger sub-pixel correction at fixed amplitude.
        # Evaluated above the normalization anchor (ℓ_norm = 2500): below the
        # anchor the ratio inverts because the wider beam dies faster at ℓ_norm
        # than at the test ℓ.
        result_wide  = sub_pixel_power([3000], 1.0, 9.68, 9.68; ℓ_pivot=3000)
        result_narrow = sub_pixel_power([3000], 1.0, 5.02, 5.02; ℓ_pivot=3000)
        # Wider beam suppresses more → larger 1/B² → larger result
        @test result_wide[1] > result_narrow[1]
    end

    # ---------------------------------------------------------
    # cib_clustered_template_power
    # ---------------------------------------------------------
    @testset "cib_clustered_template_power" begin
        tmpl = collect(range(0.5, 1.5, length=length(ℓs)))
        Acib, β, Tdust, ν0_cib = 4.0, 1.75, 25.0, 143.0
        ν1, ν2 = 147.5, 228.1   # Hillipop effective 143, 217

        result = cib_clustered_template_power(tmpl, Acib, β, Tdust, ν0_cib, ν1, ν2)
        @test length(result) == length(tmpl)
        @test all(isfinite.(result))
        @test all(result .> 0.0)

        # At ν=ν0=ν_ref, SED ratio = 1 → result = A * template
        res_ref = cib_clustered_template_power(tmpl, Acib, β, Tdust, ν0_cib, ν0_cib, ν0_cib)
        @test all(isapprox.(res_ref, Acib .* tmpl; rtol=1e-12))

        # Amplitude scaling
        result2 = cib_clustered_template_power(tmpl, 2 * Acib, β, Tdust, ν0_cib, ν1, ν2)
        @test all(isapprox.(result2 ./ result, 2.0; rtol=1e-12))

        # Equivalence with the inline form: A * s1 * s2 .* template
        s1 = CMBForegrounds.cib_mbb_sed_weight(β, Tdust, ν0_cib, ν1)
        s2 = CMBForegrounds.cib_mbb_sed_weight(β, Tdust, ν0_cib, ν2)
        expected = @. (Acib * s1 * s2) * tmpl
        @test all(isapprox.(result, expected; rtol=1e-12))

        # Symmetry under (ν1, ν2) swap
        result_swapped = cib_clustered_template_power(tmpl, Acib, β, Tdust, ν0_cib, ν2, ν1)
        @test all(isapprox.(result, result_swapped; rtol=1e-12))
    end

    # ---------------------------------------------------------
    # tsz_cib_template_power
    # ---------------------------------------------------------
    @testset "tsz_cib_template_power" begin
        tmpl = collect(range(0.1, 0.6, length=length(ℓs)))
        ξ, A_tSZ, A_CIB = 0.1, 5.0, 4.0
        β, Tdust = 1.75, 25.0
        ν0_tsz, ν0_cib = 143.0, 143.0
        ν_sz1, ν_sz2 = 100.24, 222.044
        ν_cib1, ν_cib2 = 105.2, 228.1

        result = tsz_cib_template_power(tmpl, ξ, A_tSZ, A_CIB, β, Tdust,
                                         ν0_tsz, ν0_cib,
                                         ν_sz1, ν_sz2, ν_cib1, ν_cib2)
        @test length(result) == length(tmpl)
        @test all(isfinite.(result))

        # ξ=0 → identically zero
        result_xi0 = tsz_cib_template_power(tmpl, 0.0, A_tSZ, A_CIB, β, Tdust,
                                             ν0_tsz, ν0_cib,
                                             ν_sz1, ν_sz2, ν_cib1, ν_cib2)
        @test all(result_xi0 .== 0.0)

        # ξ scales linearly
        result_2xi = tsz_cib_template_power(tmpl, 2 * ξ, A_tSZ, A_CIB, β, Tdust,
                                             ν0_tsz, ν0_cib,
                                             ν_sz1, ν_sz2, ν_cib1, ν_cib2)
        @test all(isapprox.(result_2xi ./ result, 2.0; rtol=1e-12))

        # Equivalence with the inline form
        g1 = CMBForegrounds.tsz_g_ratio(ν_sz1, ν0_tsz, CMBForegrounds.T_CMB)
        g2 = CMBForegrounds.tsz_g_ratio(ν_sz2, ν0_tsz, CMBForegrounds.T_CMB)
        s1 = CMBForegrounds.cib_mbb_sed_weight(β, Tdust, ν0_cib, ν_cib1)
        s2 = CMBForegrounds.cib_mbb_sed_weight(β, Tdust, ν0_cib, ν_cib2)
        factor = -ξ * sqrt(A_CIB * A_tSZ) * (g1 * s2 + g2 * s1)
        expected = factor .* tmpl
        @test all(isapprox.(result, expected; rtol=1e-12))

        # Symmetry under simultaneous (sz, cib) leg swap
        result_swap = tsz_cib_template_power(tmpl, ξ, A_tSZ, A_CIB, β, Tdust,
                                              ν0_tsz, ν0_cib,
                                              ν_sz2, ν_sz1, ν_cib2, ν_cib1)
        @test all(isapprox.(result, result_swap; rtol=1e-12))

        # √(A·A) = sign-correct factor: doubling both A_tSZ and A_CIB → ×2
        result_2A = tsz_cib_template_power(tmpl, ξ, 2 * A_tSZ, 2 * A_CIB, β, Tdust,
                                            ν0_tsz, ν0_cib,
                                            ν_sz1, ν_sz2, ν_cib1, ν_cib2)
        @test all(isapprox.(result_2A ./ result, 2.0; rtol=1e-12))
    end
end
