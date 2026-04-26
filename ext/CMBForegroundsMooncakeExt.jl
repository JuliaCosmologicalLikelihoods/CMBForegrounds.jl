"""
    CMBForegroundsMooncakeExt

Package extension that registers Mooncake reverse-mode rules for the
cross-spectrum kernels in CMBForegrounds.

Activated automatically when both `CMBForegrounds` and `Mooncake` are
loaded in the same Julia session. Uses `@from_chainrules` to lift the
ChainRulesCore pullbacks defined in `src/rrules.jl` into Mooncake.
"""
module CMBForegroundsMooncakeExt

using CMBForegrounds: factorized_cross, factorized_cross_te, correlated_cross,
                      assemble_TT, assemble_EE, assemble_TE
using Mooncake: @from_chainrules, MinimalCtx

@from_chainrules MinimalCtx Tuple{typeof(factorized_cross),    Vector{Float64}, Vector{Float64}}
@from_chainrules MinimalCtx Tuple{typeof(factorized_cross_te), Vector{Float64}, Vector{Float64}, Vector{Float64}}
@from_chainrules MinimalCtx Tuple{typeof(correlated_cross),    Matrix{Float64}, Array{Float64,3}}
@from_chainrules MinimalCtx Tuple{typeof(assemble_TT),
    Float64, Float64, Float64,
    Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64},
    Vector{Float64}, Vector{Float64},
    Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64},
    Vector{Float64}, Vector{Float64}, Vector{Float64}}
@from_chainrules MinimalCtx Tuple{typeof(assemble_EE),
    Float64, Float64,
    Vector{Float64}, Vector{Float64},
    Vector{Float64}, Vector{Float64}}
@from_chainrules MinimalCtx Tuple{typeof(assemble_TE),
    Float64, Float64,
    Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64},
    Vector{Float64}, Vector{Float64}}

end # module CMBForegroundsMooncakeExt
