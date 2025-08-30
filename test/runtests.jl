using Test
using CMBForegrounds
using Random

@testset "CMBForegrounds" begin
    include("test_tsz_physics.jl")
    include("test_bnu_ratio.jl")
    include("test_dbdt_ratio.jl")
    include("test_tsz_g_ratio.jl")
    include("test_cib_mbb_sed_weight.jl")
    include("test_dust_tt_power_law.jl")
    include("test_cib_clustered_power.jl")
    include("test_tsz_cross_power.jl")
    include("test_tsz_cib_cross_power.jl")
    include("test_ksz_template_scaled.jl")
    include("test_dCl_dell_from_Dl.jl")
    include("test_ssl_response.jl")
    include("test_aberration_response.jl")
    include("test_cross_calibration_mean.jl")
    include("test_shot_noise_power.jl")
    include("test_gaussian_beam_window.jl")
    include("test_fwhm_arcmin_to_sigma_rad.jl")
end
