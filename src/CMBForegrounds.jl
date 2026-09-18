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

Planck's constant in J⋅s (6.62607015×10⁻³⁴ J⋅s, CODATA 2018 exact value).
"""
const h = 6.62607015e-34  # Planck's constant (CODATA 2018)

"""
    kB

Boltzmann constant in J/K (1.380649×10⁻²³ J/K, CODATA 2018 exact value).
"""
const kB = 1.380649e-23  # Boltzmann constant (CODATA 2018)

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
include("angular.jl")
include("bandpass.jl")
include("sed.jl")
include("correlation.jl")
include("cross.jl")
include("instrument.jl")
include("rrules.jl")

# Export the main functions that we want users to access
export dimensionless_freq_vars, Bnu_ratio, dBdT_ratio, tsz_g_ratio, cib_mbb_sed_weight, dust_tt_power_law, cib_clustered_power, cib_clustered_template_power, tsz_cross_power, tsz_cib_cross_power, tsz_cib_template_power, ksz_template_scaled, dCl_dell_from_Dl, ssl_response, aberration_response, cross_calibration_mean, shot_noise_power, gaussian_beam_window, fwhm_arcmin_to_sigma_rad, dust_model_template_power, radio_ps_power, dusty_ps_power, sub_pixel_power
export eval_template, eval_template_tilt, eval_powerlaw
export AbstractAngularModel, PowerLawShape, PoissonShape, TemplateShape, TiltedTemplateShape, angular_power
export AbstractSED, ModifiedBlackbodySED, RadioSED, ThermalSZSED, ConstantSED, NoSED, SkyComponent, sed_weight, eval_component, eval_component_te
export AbstractCorrelationModel, TemplateCorrelation, GeometricMeanCorrelation, correlation_power
export x_cmb, rj2cmb, cmb2bb, tsz_f, tsz_sed, mbb_sed, radio_sed, constant_sed
export trapz, RawBand, Band, AbstractBand, DeltaBand, ChromaticBeam,
       PreparedChromaticBandpass, make_band, point_band, shift_and_normalize,
       prepare_chromatic_bandpass, integrate_sed, integrate_tsz, eval_sed_bands,
       integrate_chromatic_sed, eval_chromatic_sed_bands,
       prepare_fixed_chromatic_bandpass, eval_fixed_chromatic_sed_bands
export factorized_cross, factorized_cross_te, correlated_cross, build_szxcib_cl,
       assemble_TT, assemble_EE, assemble_TE
export calibration_factor, apply_calibration,
       window_convolution,
       additive_template, add_template,
       te_leakage, et_leakage, ee_leakage, apply_te_leakage, apply_ee_leakage,
       beam_eigenmode_response, beam_eigenmode_cross,
       apply_ssl, apply_aberration

end # module CMBForegrounds
