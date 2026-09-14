"""
    test_angular_dispatch.jl

Tests for the angular representation family:
  PowerLawShape, PoissonShape, TemplateShape, TiltedTemplateShape,
  and the unified angular_power dispatch interface.
"""

using Test
using CMBForegrounds
using JET
using ADTypes
import DifferentiationInterface as DI
using ForwardDiff
using Mooncake
using Zygote
using Random

@testset "Angular Dispatch Interface" begin
    rng = MersenneTwister(42)
    ells = 2:2500
    ells_vec = collect(ells)
    
    # ----------------------------------------------------------------- #
    # 1. PowerLawShape                                                  #
    # ----------------------------------------------------------------- #
    @testset "PowerLawShape" begin
        p_shape = PowerLawShape(3000.0)
        amp = 4.5
        alpha = -0.7

        res = angular_power(p_shape, ells_vec, alpha; amp=amp)
        ref = eval_powerlaw(ells_vec, 3000.0, alpha; amp=amp)
        @test res ≈ ref
        @test angular_power(p_shape, ells_vec; amp=amp, alpha=alpha) ≈ ref

        # Zero slope -> flat amplitude
        @test all(angular_power(p_shape, ells_vec, 0.0; amp=amp) .≈ amp)

        # Type stability
        JET.@test_opt angular_power(p_shape, ells_vec, alpha; amp=amp)

        # AD wrt amp and alpha
        g_amp(a) = sum(angular_power(p_shape, ells_vec, alpha; amp=a[1]))
        g_alpha(al) = sum(angular_power(p_shape, ells_vec, al[1]; amp=amp))

        for (fn, x0) in [(g_amp, [amp]), (g_alpha, [alpha])]
            grad_fd = DI.gradient(fn, AutoForwardDiff(), x0)
            grad_mk = DI.gradient(fn, AutoMooncake(; config=nothing), x0)
            grad_zg = DI.gradient(fn, AutoZygote(), x0)
            @test grad_fd ≈ grad_mk rtol=1e-8
            @test grad_fd ≈ grad_zg rtol=1e-8
        end
    end

    # ----------------------------------------------------------------- #
    # 2. PoissonShape & Comparison with ell^2 power law                 #
    # ----------------------------------------------------------------- #
    @testset "PoissonShape & Finite-ell differences" begin
        poisson = PoissonShape(3000.0)
        p_ell2 = PowerLawShape(3000.0)
        amp = 10.0

        res_poisson = angular_power(poisson, ells_vec; amp=amp)
        res_ell2    = angular_power(p_ell2, ells_vec, 2.0; amp=amp)

        # Check exact formula
        norm = 3000.0 * 3001.0
        ref = @. amp * (ells_vec * (ells_vec + 1)) / norm
        @test res_poisson ≈ ref

        # Check finite-ell difference: at low ell, ell(ell+1) != ell^2
        # At ell=2: 2*3/(3000*3001) vs 4/(3000^2)
        # Ratio: (6/9003000) / (4/9000000) = (6/4) * (9000/9003) = 1.5 * 0.99967 ≈ 1.4995 (~50% difference!)
        @test res_poisson[1] / res_ell2[1] ≈ 1.5 atol=1e-3
        # At ell=3000: 3000*3001 / (3000*3001) = 1, while (3000/3000)^2 = 1
        idx_3000 = findfirst(==(3000), ells_vec)
        if idx_3000 !== nothing
            @test res_poisson[idx_3000] ≈ res_ell2[idx_3000] ≈ amp
        end

        # Type stability
        JET.@test_opt angular_power(poisson, ells_vec; amp=amp)

        # AD
        g_poisson(a) = sum(angular_power(poisson, ells_vec; amp=a[1]))
        grad_fd = DI.gradient(g_poisson, AutoForwardDiff(), [amp])
        grad_mk = DI.gradient(g_poisson, AutoMooncake(; config=nothing), [amp])
        grad_zg = DI.gradient(g_poisson, AutoZygote(), [amp])
        @test grad_fd ≈ grad_mk rtol=1e-8
        @test grad_fd ≈ grad_zg rtol=1e-8
    end

    # ----------------------------------------------------------------- #
    # 3. TemplateShape                                                  #
    # ----------------------------------------------------------------- #
    @testset "TemplateShape (normalized, pre-normalized, nonzero ell_min)" begin
        raw_template = abs.(randn(rng, 3501)) .+ 0.1  # ell from 0 to 3500

        # 3a. Normalized template (ell_min = 0, ell_0 = 3000)
        t_norm = TemplateShape(raw_template; ell_0=3000, ell_min=0)
        amp = 2.0
        res_norm = angular_power(t_norm, ells_vec; amp=amp)
        ref_norm = eval_template(raw_template, ells_vec, 3000; amp=amp)
        @test res_norm ≈ ref_norm
        # Check value at ell=3000 is exactly amp
        @test angular_power(t_norm, [3000]; amp=amp)[1] ≈ amp

        # 3b. Pre-normalized template (ell_0 = nothing)
        prenorm_template = raw_template ./ raw_template[3001]
        t_prenorm = TemplateShape(prenorm_template; ell_0=nothing, ell_min=0)
        res_prenorm = angular_power(t_prenorm, ells_vec; amp=amp)
        @test res_prenorm ≈ res_norm

        # 3c. Non-zero starting multipole (e.g. ell_min = 2)
        template_from2 = raw_template[3:end]  # starts at ell=2, length 3499
        t_from2 = TemplateShape(template_from2; ell_0=3000, ell_min=2)
        res_from2 = angular_power(t_from2, ells_vec; amp=amp)
        @test res_from2 ≈ res_norm

        # 3d. Omitted ell argument evaluates full template
        res_full = angular_power(t_prenorm; amp=amp)
        @test length(res_full) == length(prenorm_template)
        @test res_full ≈ amp .* prenorm_template

        # Type stability
        JET.@test_opt angular_power(t_norm, ells_vec; amp=amp)
        JET.@test_opt angular_power(t_prenorm, ells_vec; amp=amp)
        JET.@test_opt angular_power(t_from2, ells_vec; amp=amp)

        # AD wrt amp
        g_tmpl(a) = sum(angular_power(t_norm, ells_vec; amp=a[1]))
        grad_fd = DI.gradient(g_tmpl, AutoForwardDiff(), [amp])
        grad_mk = DI.gradient(g_tmpl, AutoMooncake(; config=nothing), [amp])
        grad_zg = DI.gradient(g_tmpl, AutoZygote(), [amp])
        @test grad_fd ≈ grad_mk rtol=1e-8
        @test grad_fd ≈ grad_zg rtol=1e-8

        # Query semantics do not depend on request length or ordering.
        short = TemplateShape([10.0, 20.0, 30.0]; ell_0=3, ell_min=2)
        @test angular_power(short, [4, 3, 2]) == [1.5, 1.0, 0.5]
        @test angular_power(short, [2, 4]) == [0.5, 1.5]
        @test_throws ArgumentError angular_power(short, [2.5])
        @test_throws BoundsError angular_power(short, [5])

        # Template values may be active parameters, including the pivot value.
        g_values(x) = sum(angular_power(TemplateShape(x; ell_0=2, ell_min=2), [2, 3]))
        expected = [-2.0, 1.0]
        @test DI.gradient(g_values, AutoForwardDiff(), [1.0, 2.0]) ≈ expected
        @test DI.gradient(g_values, AutoMooncake(; config=nothing), [1.0, 2.0]) ≈ expected
        @test DI.gradient(g_values, AutoZygote(), [1.0, 2.0]) ≈ expected
    end

    # ----------------------------------------------------------------- #
    # 4. TiltedTemplateShape                                            #
    # ----------------------------------------------------------------- #
    @testset "TiltedTemplateShape" begin
        raw_template = abs.(randn(rng, 3501)) .+ 0.1
        amp = 1.8
        alpha = -0.5
        ell_0 = 3000

        # Constructed from template array
        tilted = TiltedTemplateShape(raw_template; ell_0=ell_0, ell_min=0)
        res = angular_power(tilted, ells_vec, alpha; amp=amp)
        # Note: raw_template is unnormalized, but base in constructor has ell_0=nothing, so rescales raw_template directly
        ref = amp .* raw_template[ells_vec .+ 1] .* (ells_vec ./ ell_0).^alpha
        @test res ≈ ref

        # Constructed from normalized TemplateShape
        base_norm = TemplateShape(raw_template; ell_0=ell_0, ell_min=0)
        tilted_norm = TiltedTemplateShape(base_norm, ell_0)
        res_norm = angular_power(tilted_norm, ells_vec, alpha; amp=amp)
        ref_norm = eval_template_tilt(raw_template, ells_vec, ell_0, alpha; amp=amp)
        @test res_norm ≈ ref_norm

        # Type stability
        JET.@test_opt angular_power(tilted_norm, ells_vec, alpha; amp=amp)

        # AD wrt amp and alpha
        g_tilted(p) = sum(angular_power(tilted_norm, ells_vec, p[2]; amp=p[1]))
        p0 = [amp, alpha]
        grad_fd = DI.gradient(g_tilted, AutoForwardDiff(), p0)
        grad_mk = DI.gradient(g_tilted, AutoMooncake(; config=nothing), p0)
        grad_zg = DI.gradient(g_tilted, AutoZygote(), p0)
        @test grad_fd ≈ grad_mk rtol=1e-8
        @test grad_fd ≈ grad_zg rtol=1e-8

        # The derivative at zero tilt is not zero.
        base = TemplateShape([2.0, 3.0]; ell_0=nothing, ell_min=2)
        zero_tilt = TiltedTemplateShape(base, 2.0)
        g_zero(p) = sum(angular_power(zero_tilt, [2, 3]; alpha=p[1]))
        expected_zero = [3 * log(3 / 2)]
        @test DI.gradient(g_zero, AutoForwardDiff(), [0.0]) ≈ expected_zero
        @test DI.gradient(g_zero, AutoMooncake(; config=nothing), [0.0]) ≈ expected_zero
        @test DI.gradient(g_zero, AutoZygote(), [0.0]) ≈ expected_zero
    end

    # ----------------------------------------------------------------- #
    # 5. Polymorphic Interchangeability                                  #
    # ----------------------------------------------------------------- #
    @testset "Interchangeability in foreground component path" begin
        raw_template = abs.(randn(rng, 3501)) .+ 0.1
        norm_template = raw_template ./ raw_template[3001]
        
        # Two alternative representations for CIB angular shape:
        shape_pl = PowerLawShape(3000.0)
        shape_tm = TemplateShape(norm_template; ell_0=nothing, ell_min=0)

        # Mock generic component evaluator
        function eval_component(angular_model::AbstractAngularModel, ells, sed_factor, amp; kwargs...)
            ang = angular_power(angular_model, ells; amp=amp, kwargs...)
            return sed_factor .* ang
        end

        sed_factor = 2.3
        amp = 5.0
        slope = 0.8

        c_pl = eval_component(shape_pl, ells_vec, sed_factor, amp; alpha=slope)
        c_tm = eval_component(shape_tm, ells_vec, sed_factor, amp)

        @test length(c_pl) == length(ells_vec)
        @test length(c_tm) == length(ells_vec)
        @test c_pl ≈ sed_factor * amp .* (ells_vec ./ 3000.0).^slope
        @test c_tm ≈ sed_factor * amp .* norm_template[ells_vec .+ 1]
    end
end
