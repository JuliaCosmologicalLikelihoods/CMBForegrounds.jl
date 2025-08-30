using Test

# Run critical tSZ physics validation tests
println("🧪 Running CMBForegrounds.jl Test Suite")
println("="^50)

include("test_tsz_physics.jl")
include("test_bnu_ratio.jl")
include("test_dbdt_ratio.jl")
include("test_tsz_g_ratio.jl")
