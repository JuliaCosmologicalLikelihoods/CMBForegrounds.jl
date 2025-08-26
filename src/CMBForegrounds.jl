module CMBForegrounds

const T_CMB = 2.72548  # CMB temperature
const h = 6.62606957e-34  # Planck's constant
const kB = 1.3806488e-23  # Boltzmann constant
const Ghz_Kelvin = h / kB * 1e9

const galdust_ν0 = 150
const galdust_T = 19.6
const CIB_ν0 = 150.0
const CIB_T = 25.0
const tSZ_ν0 = 143

include("foregrounds.jl")

end # module CMBForegrounds
