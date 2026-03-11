"""
    dimensionless_freq_vars(ν, ν0, T)

Calculates dimensionless frequency variables used in blackbody-related calculations.

The variables are defined as:
- ``r = \\nu / \\nu_0``
- ``x = \\frac{h\\nu}{k_B T}``
- ``x_0 = \\frac{h\\nu_0}{k_B T}``

where ``h`` is the Planck constant and ``k_B`` is the Boltzmann constant.

# Arguments
- `ν`: Frequency in GHz
- `ν0`: Reference frequency in GHz
- `T`: Temperature in Kelvin

# Returns
- A tuple `(r, x, x0)` containing the dimensionless variables
"""
function dimensionless_freq_vars(ν, ν0, T)
    νu, ν0u, Tu = promote(ν, ν0, T)
    r = νu / ν0u
    x0 = Ghz_Kelvin * ν0u / Tu
    x = r * x0
    return r, x, x0
end

"""
    Bnu_ratio(ν, ν0, T)

Computes the ratio of Planck blackbody intensities at two frequencies.

The Planck function ratio is:
```math
\\frac{B(\\nu, T)}{B(\\nu_0, T)} = \\left(\\frac{\\nu}{\\nu_0}\\right)^3 \\frac{\\exp(h\\nu_0/k_B T) - 1}{\\exp(h\\nu/k_B T) - 1}
```

# Arguments
- `ν`: Frequency in GHz
- `ν0`: Reference frequency in GHz
- `T`: Temperature in Kelvin

# Returns
- Dimensionless ratio `B(ν, T) / B(ν0, T)`
"""
function Bnu_ratio(ν, ν0, T)
    r, x, x0 = dimensionless_freq_vars(ν, ν0, T)
    return r^3 * expm1(x0) / expm1(x)
end

"""
    dBdT_ratio(ν, ν0, T)

Computes the ratio of blackbody temperature derivatives at two frequencies.

The derivative ratio is:
```math
\\frac{\\partial B(\\nu, T)/\\partial T}{\\partial B(\\nu_0, T)/\\partial T} = \\left(\\frac{\\nu}{\\nu_0}\\right)^4 \\frac{x_0^2 \\exp(x_0)/(\\exp(x_0) - 1)^2}{x^2 \\exp(x)/(\\exp(x) - 1)^2}
```

where ``x = h\\nu/(k_B T)`` and ``x_0 = h\\nu_0/(k_B T)``.

# Arguments
- `ν`: Frequency in GHz
- `ν0`: Reference frequency in GHz
- `T`: Temperature in Kelvin

# Returns
- Dimensionless ratio `(∂B/∂T)(ν, T) / (∂B/∂T)(ν0, T)`
"""
function dBdT_ratio(ν, ν0, T)
    r, x, x0 = dimensionless_freq_vars(ν, ν0, T)

    # use exp(x)/(exp(x)-1)^2 = 1/(4*sinh(x/2)^2)
    s0 = sinh(x0 / 2)
    s = sinh(x / 2)

    return r^4 * (s0 * s0) / (s * s)
end

"""
    tsz_g_ratio(ν, ν0, T)

Calculates the spectral shape of the thermal Sunyaev-Zel'dovich (tSZ) effect.

The tSZ spectral function is:
```math
g(x) = x \\coth\\left(\\frac{x}{2}\\right) - 4
```

where ``x = h\\nu/(k_B T)``. This function returns ``g(x) / g(x_0)``.

# Arguments
- `ν`: Frequency in GHz
- `ν0`: Reference frequency in GHz
- `T`: CMB temperature in Kelvin

# Returns
- Dimensionless ratio `g(ν) / g(ν0)` of tSZ spectral function
"""
function tsz_g_ratio(ν, ν0, T)
    r, x, x0 = dimensionless_freq_vars(ν, ν0, T)
    g0 = x0 * (one(x0) + 2 / expm1(x0)) - 4
    g = x * (one(x) + 2 / expm1(x)) - 4
    return g / g0
end

"""
    cib_mbb_sed_weight(β, Tdust, ν0, ν; T_CMB=T_CMB)

Calculates the spectral energy distribution weight for modified blackbody emission.

The CIB modified blackbody SED weight is:
```math
S(\\nu) = \\left(\\frac{\\nu}{\\nu_0}\\right)^\\beta \\frac{B(\\nu, T_\\mathrm{dust})}{B(\\nu_0, T_\\mathrm{dust})} \\frac{(\\partial B/\\partial T)(\\nu_0, T_\\mathrm{CMB})}{(\\partial B/\\partial T)(\\nu, T_\\mathrm{CMB})}
```

where ``B(\\nu, T)`` is the Planck function and ``\\beta`` is the dust emissivity index.

# Arguments
- `β`: Dust emissivity spectral index
- `Tdust`: Dust temperature in Kelvin
- `ν0`: Reference frequency in GHz
- `ν`: Evaluation frequency in GHz

# Keywords
- `T_CMB=T_CMB`: CMB temperature in Kelvin

# Returns
- Dimensionless SED weight
"""
function cib_mbb_sed_weight(β, Tdust, ν0, ν; T_CMB=T_CMB)
    βu, Tdu, ν0u, νu, Tcu = promote(β, Tdust, ν0, ν, T_CMB)
    r = νu / ν0u
    return r^βu * Bnu_ratio(νu, ν0u, Tdu) / dBdT_ratio(νu, ν0u, Tcu)
end

"""
    dust_tt_power_law(ℓs, A_pivot, α, β, ν1, ν2, Tdust, ν0; ℓ_pivot=80, T_CMB=T_CMB)

Computes thermal dust power spectrum using a power-law model.

The dust power spectrum is:
```math
D_\\ell^{\\mathrm{dust}}(\\nu_1, \\nu_2) = A_{\\mathrm{pivot}} \\cdot S(\\nu_1) \\cdot S(\\nu_2) \\cdot \\left(\\frac{\\ell}{\\ell_{\\mathrm{pivot}}}\\right)^{\\alpha + 2}
```

where ``S(\\nu)`` is the modified blackbody SED weight and the ``+2`` converts from ``C_\\ell`` to ``D_\\ell`` scaling.

# Arguments
- `ℓs`: Multipoles vector
- `A_pivot`: Amplitude at pivot multipole
- `α`: Power-law index for multipole dependence
- `β`: Dust emissivity spectral index
- `ν1`, `ν2`: Frequencies in GHz for cross-correlation
- `Tdust`: Dust temperature in Kelvin
- `ν0`: Reference frequency in GHz

# Keywords
- `ℓ_pivot=80`: Pivot multipole
- `T_CMB=T_CMB`: CMB temperature

# Returns
- Dust power spectrum `Dℓ`
"""
function dust_tt_power_law(ℓs::AbstractVector, A_pivot, α, β, ν1, ν2, Tdust, ν0;
    ℓ_pivot=80, T_CMB=T_CMB)
    s1 = cib_mbb_sed_weight(β, Tdust, ν0, ν1; T_CMB=T_CMB)
    s2 = cib_mbb_sed_weight(β, Tdust, ν0, ν2; T_CMB=T_CMB)
    # Also rename A80 to A_pivot to reflect its general nature
    return (ℓs ./ ℓ_pivot) .^ (α + 2) .* (A_pivot * s1 * s2)
end

"""
    cib_clustered_power(ℓs, A_CIB, α, β, ν1, ν2, z1, z2, Tdust, ν0_cib; ℓ_pivot=3000, T_CMB=T_CMB)

Computes the clustered cosmic infrared background power spectrum.

The CIB clustered power spectrum is:
```math
D_\\ell^{\\mathrm{CIB}}(\\nu_1, \\nu_2) = A_{\\mathrm{CIB}} \\cdot S(\\nu_1) \\cdot S(\\nu_2) \\cdot \\sqrt{z_1 z_2} \\cdot \\left(\\frac{\\ell}{\\ell_{\\mathrm{pivot}}}\\right)^\\alpha
```

where ``S(\\nu)`` are the modified blackbody SED weights and ``z_i`` are redshift factors.

This function handles both auto-spectra (when ν1=ν2 and z1=z2) and cross-spectra.

# Arguments
- `ℓs`: An `AbstractVector` of multipoles.
- `A_CIB`: Amplitude of the CIB power spectrum at the pivot multipole.
- `α`: Power-law index for the multipole dependence.
- `β`: Spectral index for the dust emissivity (modified blackbody).
- `ν1`: First frequency in GHz.
- `ν2`: Second frequency in GHz.
- `z1`: First redshift factor (related to flux normalization).
- `z2`: Second redshift factor (related to flux normalization).
- `Tdust`: Dust temperature in Kelvin.
- `ν0_cib`: Reference frequency for the CIB SED in GHz.

# Keywords
- `ℓ_pivot`: Pivot multipole where amplitude is defined, default is 3000.
- `T_CMB`: Temperature of the CMB in Kelvin, default is T_CMB constant.

# Returns
- An `AbstractVector` containing the CIB clustered power spectrum `D_ℓ` at each `ℓ` in `ℓs`.

# Examples
```julia
# Auto-spectrum at 353 GHz
D_ℓ_auto = cib_clustered_power(ℓs, 1.0, 0.8, 1.6, 353.0, 353.0, 1.0, 1.0, 25.0, 150.0)

# Cross-spectrum between 217 and 353 GHz  
D_ℓ_cross = cib_clustered_power(ℓs, 1.0, 0.8, 1.6, 217.0, 353.0, 0.9, 1.1, 25.0, 150.0)
```
"""
function cib_clustered_power(ℓs::AbstractVector,
    A_CIB, α, β, ν1, ν2, z1, z2,
    Tdust, ν0_cib; ℓ_pivot=3000, T_CMB=T_CMB)
    s1 = cib_mbb_sed_weight(β, Tdust, ν0_cib, ν1; T_CMB=T_CMB)
    s2 = cib_mbb_sed_weight(β, Tdust, ν0_cib, ν2; T_CMB=T_CMB)

    return @. (A_CIB * s1 * s2 * sqrt(z1 * z2)) * (ℓs / ℓ_pivot)^α
end

"""
    tsz_cross_power(template, A_tSZ, ν1, ν2, ν0, α_tSZ, ℓ_pivot, ℓs; T_CMB=T_CMB)

Computes the thermal Sunyaev-Zel'dovich cross-power spectrum.

The tSZ power spectrum is:
```math
D_\\ell^{\\mathrm{tSZ}}(\\nu_1, \\nu_2) = A_{\\mathrm{tSZ}} \\cdot g(\\nu_1) \\cdot g(\\nu_2) \\cdot T(\\ell) \\cdot \\left(\\frac{\\ell}{\\ell_{\\mathrm{pivot}}}\\right)^{\\alpha_{\\mathrm{tSZ}}}
```

where ``T(\\ell)`` is the template and ``g(\\nu)`` is the tSZ spectral function.

# Arguments
- `template`: tSZ power spectrum template `Dℓ` at reference frequency
- `A_tSZ`: tSZ amplitude at pivot scale
- `ν1`, `ν2`: Frequencies in GHz of correlated channels
- `ν0`: Reference frequency in GHz
- `α_tSZ`: Power-law tilt of tSZ spectrum
- `ℓ_pivot`: Pivot multipole for power-law scaling
- `ℓs`: Multipoles for computation

# Keywords
- `T_CMB=T_CMB`: CMB temperature in Kelvin

# Returns
- tSZ cross-power spectrum `Dℓ`
"""
function tsz_cross_power(template::AbstractVector, A_tSZ, ν1, ν2, ν0, α_tSZ, ℓ_pivot, ℓs::AbstractVector; T_CMB=T_CMB)
    # Preserve AD types (Dual, BigFloat, etc.)
    A, ν1_, ν2_, ν0_, T_CMB_ = promote(A_tSZ, ν1, ν2, ν0, T_CMB)

    s1 = tsz_g_ratio(ν1_, ν0_, T_CMB_)
    s2 = tsz_g_ratio(ν2_, ν0_, T_CMB_)

    # Single-pass broadcast; result eltype promotes with A,s1,s2
    return @. template * (A * s1 * s2) * (ℓs / ℓ_pivot)^α_tSZ
end

"""
    tsz_cib_cross_power(ℓs, ξ, A_tSZ, A_CIB, α, β, z1, z2, ν_cib1, ν_cib2, ν_tsz1, ν_tsz2, α_tsz, tsz_template, ν0_tsz, Tdust, ν0_cib; ℓ_pivot_cib=3000, ℓ_pivot_tsz=3000, T_CMB=T_CMB)

Computes the cross-correlation between thermal SZ and cosmic infrared background.

The tSZ-CIB cross-power spectrum is:
```math
D_\\ell^{\\mathrm{tSZ \\times CIB}} = -\\xi \\left( \\sqrt{|D_\\ell^{\\mathrm{tSZ,11}} \\cdot D_\\ell^{\\mathrm{CIB,22}}|} + \\sqrt{|D_\\ell^{\\mathrm{tSZ,22}} \\cdot D_\\ell^{\\mathrm{CIB,11}}|} \\right)
```

where ``\\xi`` is the correlation coefficient and auto-spectra are computed for each component.

# Arguments
- `ℓs`: Vector of multipoles.
- `ξ`: tSZ-CIB correlation coefficient.
- `A_tSZ`, `A_CIB`: Amplitudes for the tSZ and CIB power spectra.
- `α`, `β`: Power-law indices for the CIB model.
- `z1`, `z2`: Redshifts for the CIB channels.
- `ν_cib1`, `ν_cib2`: Frequencies for the CIB channels.
- `ν_tsz1`, `ν_tsz2`: Frequencies for the tSZ channels.
- `α_tsz`: The power-law tilt of the tSZ power spectrum.
- `tsz_template`: Power spectrum template for the tSZ effect.
- `ν0_tsz`, `ν0_cib`: Reference frequencies for the tSZ and CIB models.
- `Tdust`: Dust temperature for the CIB model.

# Keywords
- `ℓ_pivot_cib`: Pivot multipole for the CIB power spectrum; default is 3000.
- `ℓ_pivot_tsz`: Pivot multipole for the tSZ power spectrum; default is 3000.
- `T_CMB`: Temperature of the CMB.

# Returns
- An `AbstractVector` of the tSZ-CIB cross-power spectrum `Dℓ`.
"""
function tsz_cib_cross_power(
    ℓs::AbstractVector,
    ξ, A_tSZ, A_CIB, α, β, z1, z2,
    ν_cib1, ν_cib2, ν_tsz1, ν_tsz2, α_tsz,
    tsz_template::AbstractVector,
    ν0_tsz, Tdust, ν0_cib; ℓ_pivot_cib=3000, ℓ_pivot_tsz=3000, T_CMB=T_CMB
)
    @assert length(ℓs) == length(tsz_template)

    # CIB autos
    cib_11 = cib_clustered_power(ℓs, A_CIB, α, β, ν_cib1, ν_cib1, z1, z1, Tdust, ν0_cib; ℓ_pivot=ℓ_pivot_cib, T_CMB=T_CMB)
    cib_22 = cib_clustered_power(ℓs, A_CIB, α, β, ν_cib2, ν_cib2, z2, z2, Tdust, ν0_cib; ℓ_pivot=ℓ_pivot_cib, T_CMB=T_CMB)

    # tSZ autos
    tsz_11 = tsz_cross_power(tsz_template, A_tSZ, ν_tsz1, ν_tsz1, ν0_tsz, α_tsz, ℓ_pivot_tsz, ℓs; T_CMB=T_CMB)
    tsz_22 = tsz_cross_power(tsz_template, A_tSZ, ν_tsz2, ν_tsz2, ν0_tsz, α_tsz, ℓ_pivot_tsz, ℓs; T_CMB=T_CMB)

    return @. -ξ * (sqrt(abs(tsz_11 * cib_22)) + sqrt(abs(tsz_22 * cib_11)))
end

"""
    ksz_template_scaled(template, AkSZ)

Scales a kinematic Sunyaev-Zel'dovich (kSZ) power spectrum template by a given amplitude.

# Arguments
- `template`: An `AbstractVector` representing the kSZ power spectrum shape `Dℓ`.
- `AkSZ`: The amplitude scaling factor.

# Returns
- An `AbstractVector` containing the scaled kSZ power spectrum.
"""
function ksz_template_scaled(template::AbstractVector, AkSZ)
    # Broadcasted multiply promotes types automatically (e.g., Dual, BigFloat)
    return @. template * AkSZ
end

"""
    dCl_dell_from_Dl(ℓs, Dℓ)

Calculates the derivative of the angular power spectrum.

Converts from ``D_\\ell`` to ``C_\\ell`` derivative using:
```math
\\frac{\\mathrm{d}C_\\ell}{\\mathrm{d}\\ell} = \\frac{\\mathrm{d}}{\\mathrm{d}\\ell}\\left[D_\\ell \\frac{2\\pi}{\\ell(\\ell+1)}\\right]
```

The derivative is computed using central differences for interior points.

# Arguments
- `ℓs`: Multipoles vector
- `Dℓ`: Power spectrum `D_\\ell` values

# Returns
- Derivative `dCℓ/dℓ`
"""
function dCl_dell_from_Dl(ℓs::AbstractVector, Dℓ::AbstractVector)
    @assert length(ℓs) == length(Dℓ) "ells and Dℓ must have the same length"
    n = length(Dℓ)
    @assert n ≥ 2 "Need at least two multipoles to form a derivative"

    # Convert Dℓ → Cℓ without forcing Float64; promotion happens automatically.
    Cℓ = @. Dℓ * (2π) / (ℓs * (ℓs + 1))

    dCℓ = similar(Cℓ)

    @inbounds begin
        # Central differences for interior points
        for i in 2:n-1
            dCℓ[i] = (Cℓ[i+1] - Cℓ[i-1]) / (ℓs[i+1] - ℓs[i-1])
        end

        dCℓ[1] = dCℓ[2]
        dCℓ[n] = dCℓ[n-1]
    end

    return dCℓ
end

"""
    ssl_response(ℓs, κ, Dℓ)

Calculates the super-sample lensing response in the power spectrum.

The SSL response is:
```math
\\Delta D_\\ell^{\\mathrm{SSL}} = -\\kappa \\left[ \\ell \\frac{\\ell(\\ell+1)}{2\\pi} \\frac{\\mathrm{d}C_\\ell}{\\mathrm{d}\\ell} + 2 D_\\ell \\right]
```

where ``\\kappa`` is the convergence field and ``C_\\ell = D_\\ell \\cdot 2\\pi/[\\ell(\\ell+1)]``.

# Arguments
- `ℓs`: Multipoles vector
- `κ`: Convergence field value
- `Dℓ`: Unperturbed power spectrum

# Returns
- Change in power spectrum due to SSL
"""
function ssl_response(ℓs::AbstractVector, κ, Dℓ::AbstractVector)
    @assert length(ℓs) == length(Dℓ) "ells and Dℓ must have the same length"

    dCℓ = dCl_dell_from_Dl(ℓs, Dℓ)

    out = similar(Dℓ)
    @inbounds @simd for i in eachindex(out, ℓs, dCℓ, Dℓ)
        ℓ = ℓs[i]
        pref = (ℓ * ℓ * (ℓ + 1)) / (2π)  # ℓ * ℓ(ℓ+1)/(2π)
        ssl = pref * dCℓ[i] + 2 * Dℓ[i]
        out[i] = -κ * ssl
    end
    return out
end

"""
    aberration_response(ℓs, ab_coeff, Dℓ)

Calculates the relativistic aberration response in the power spectrum.

The aberration response is:
```math
\\Delta D_\\ell^{\\mathrm{aberr}} = -A_{\\mathrm{aberr}} \\cdot \\ell \\frac{\\ell(\\ell+1)}{2\\pi} \\frac{\\mathrm{d}C_\\ell}{\\mathrm{d}\\ell}
```

where ``A_{\\mathrm{aberr}}`` is the aberration coefficient related to observer velocity.

# Arguments
- `ℓs`: Multipoles vector
- `ab_coeff`: Aberration coefficient
- `Dℓ`: Unperturbed power spectrum

# Returns
- Change in power spectrum due to aberration
"""
function aberration_response(ℓs::AbstractVector, ab_coeff, Dℓ::AbstractVector)
    @assert length(ℓs) == length(Dℓ) "ells and Dℓ must have the same length"
    dCℓ = dCl_dell_from_Dl(ℓs, Dℓ)

    out = similar(Dℓ)
    @inbounds @simd for i in eachindex(out, ℓs, dCℓ, Dℓ)
        ℓ = ℓs[i]
        pref = (ℓ * ℓ * (ℓ + 1)) / (2π)   # ℓ * ℓ(ℓ+1)/(2π)
        out[i] = -ab_coeff * dCℓ[i] * pref
    end
    return out
end

"""
    cross_calibration_mean(cal1, cal2, cal3, cal4)

Calculates the mean of two cross-calibration products. This is a simple utility
function, often used for combining calibration factors from different frequency maps.

# Arguments
- `cal1`, `cal2`, `cal3`, `cal4`: Calibration factors.

# Returns
- The mean value: `(cal1 * cal2 + cal3 * cal4) / 2`.
"""
@inline function cross_calibration_mean(cal1, cal2, cal3, cal4)
    return (cal1 * cal2 + cal3 * cal4) / 2
end

"""
    shot_noise_power(ℓs, A_ℓ0; ℓ0=3000)

Computes shot noise power spectrum with ℓ² scaling.

The shot noise power spectrum is:
```math
D_\\ell^{\\mathrm{shot}} = A_{\\ell_0} \\left(\\frac{\\ell}{\\ell_0}\\right)^2
```

This represents the white noise contribution from point sources.

# Arguments
- `ℓs`: Multipoles vector
- `A_ℓ0`: Shot noise amplitude at reference multipole

# Keywords
- `ℓ0=3000`: Reference multipole

# Returns
- Shot noise power spectrum `Dℓ`
"""
function shot_noise_power(ℓs::AbstractVector, A_ℓ0; ℓ0=3000)
    s = A_ℓ0 / (ℓ0 * ℓ0)
    return @. ℓs * ℓs * s
end

"""
    fwhm_arcmin_to_sigma_rad(fwhm_arcmin)

Converts a beam's Full-Width at Half-Maximum (FWHM) from arcminutes to its
standard deviation (sigma) in radians.

# Arguments
- `fwhm_arcmin`: The FWHM in units of arcminutes.

# Returns
- The beam sigma in units of radians.
"""
@inline fwhm_arcmin_to_sigma_rad(fwhm_arcmin) =
    (fwhm_arcmin * (π / 180) / 60) / sqrt(8 * log(2))

"""
    gaussian_beam_window(fwhm_arcmin, ells)

Calculates the Gaussian beam window function.

The beam window function is:
```math
B_\\ell = \\exp\\left(-\\frac{1}{2} \\ell(\\ell + 1) \\sigma^2\\right)
```

where ``\\sigma = \\mathrm{FWHM} / \\sqrt{8 \\ln 2}`` is the beam standard deviation in radians.

# Arguments
- `fwhm_arcmin`: Beam FWHM in arcminutes
- `ells`: Multipoles vector

# Returns
- Beam window function values
"""
function gaussian_beam_window(fwhm_arcmin, ells::AbstractVector)
    σ = fwhm_arcmin_to_sigma_rad(fwhm_arcmin)
    return @. exp(-0.5 * ells * (ells + 1) * σ^2)
end
