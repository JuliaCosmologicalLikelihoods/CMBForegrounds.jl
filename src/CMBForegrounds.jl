module CMBForegrounds

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

# Export the main functions that we want users to access
export dimensionless_freq_vars, Bnu_ratio, dBdT_ratio, tsz_g_ratio, cib_mbb_sed_weight

end # module CMBForegrounds
