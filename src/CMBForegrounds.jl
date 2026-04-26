module CMBForegrounds

using ChainRulesCore
using LinearAlgebra

"""
    T_CMB

The temperature of the Cosmic Microwave Background in Kelvin (2.72548 K).
"""
const T_CMB = 2.72548  # CMB temperature

"""
    h

Planck's constant in J⋅s (6.62606957×10⁻³⁴ J⋅s).
"""
const h = 6.62606957e-34  # Planck's constant

"""
    kB

Boltzmann constant in J/K (1.3806488×10⁻²³ J/K).
"""
const kB = 1.3806488e-23  # Boltzmann constant

"""
    Ghz_Kelvin

Conversion factor h/kB × 10⁹ for converting between frequency (GHz) and temperature (K).
"""
const Ghz_Kelvin = h / kB * 1e9

const galdust_ν0 = 150
const galdust_T = 19.6
const CIB_ν0 = 150.0
const CIB_T = 25.0
const tSZ_ν0 = 143

include("foregrounds.jl")
include("bandpass.jl")
include("cross.jl")
include("rrules.jl")
include("components.jl")

# Export the main functions that we want users to access
export dimensionless_freq_vars, Bnu_ratio, dBdT_ratio, tsz_g_ratio, cib_mbb_sed_weight, dust_tt_power_law, cib_clustered_power, tsz_cross_power, tsz_cib_cross_power, ksz_template_scaled, dCl_dell_from_Dl, ssl_response, aberration_response, cross_calibration_mean, shot_noise_power, gaussian_beam_window, fwhm_arcmin_to_sigma_rad, dust_model_template_power, radio_ps_power, dusty_ps_power, sub_pixel_power
export eval_template, eval_template_tilt, eval_powerlaw
export x_cmb, rj2cmb, cmb2bb, tsz_f, tsz_sed, mbb_sed, radio_sed, constant_sed
export trapz, RawBand, Band, make_band, point_band, shift_and_normalize,
       integrate_sed, integrate_tsz, eval_sed_bands
export factorized_cross, factorized_cross_te, correlated_cross, build_szxcib_cl,
       assemble_TT, assemble_EE, assemble_TE
export FGComponent, KSZ, TSZ, CIBPoisson, CIBClustered, CorrelatedTSZxCIB,
       Radio, DustPL, TSZxCIBAuto, DustTemplate, ShotNoise, SubPixel
export FGContext, compute_dl, compute_fg_total

end # module CMBForegrounds
