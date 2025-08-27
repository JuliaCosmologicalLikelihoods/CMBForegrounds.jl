"""
    dimensionless_freq_vars(ν, ν0, T)

Calculates dimensionless frequency variables used in blackbody-related calculations.

The variables are defined as:
- `r = ν / ν0`
- `x = hν / (k_B T)`
- `x0 = hν0 / (k_B T)`

where `h` is the Planck constant and `k_B` is the Boltzmann constant.

# Arguments
- `ν`: Frequency.
- `ν0`: Reference frequency.
- `T`: Temperature in Kelvin.

# Returns
- A tuple `(r, x, x0)`.
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

Computes the ratio of the blackbody intensity `B(ν, T)` at two different frequencies.

The ratio is given by: `(ν/ν0)³ * (exp(hν0/k_B T) - 1) / (exp(hν/k_B T) - 1)`.

# Arguments
- `ν`: Frequency.
- `ν0`: Reference frequency.
- `T`: Temperature in Kelvin.

# Returns
- The dimensionless ratio `B(ν, T) / B(ν0, T)`.
"""
function Bnu_ratio(ν, ν0, T)
    r, x, x0 = dimensionless_freq_vars(ν, ν0, T)
    return r^3 * expm1(x0) / expm1(x)
end

"""
    dBdT_ratio(ν, ν0, T)

Computes the ratio of the blackbody derivative `dB/dT` at two different frequencies.

This function uses a numerically stable formula: `1 / (4 * sinh(x/2)^2)` which is
equivalent to `exp(x) / (exp(x) - 1)^2`.

# Arguments
- `ν`: Frequency.
- `ν0`: Reference frequency.
- `T`: Temperature in Kelvin.

# Returns
- The dimensionless ratio `(dB/dT)(ν, T) / (dB/dT)(ν0, T)`.
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

Calculates the spectral shape of the thermal Sunyaev-Zel'dovich (tSZ) effect,
normalized at a reference frequency `ν0`.

The tSZ spectral function is `g(x) = x * coth(x/2) - 4`, where `x = hν / (k_B T)`.
This function computes `g(x) / g(x0)`.

# Arguments
- `ν`: Frequency.
- `ν0`: Reference frequency.
- `T`: Temperature of the CMB in Kelvin.

# Returns
- The dimensionless ratio of the tSZ spectral function at `ν` and `ν0`.
"""
function tsz_g_ratio(ν, ν0, T)
    r, x, x0 = dimensionless_freq_vars(ν, ν0, T)
    g0 = x0 * (one(x0) + 2 / expm1(x0)) - 4
    g = x * (one(x) + 2 / expm1(x)) - 4
    return g / g0
end

"""
    cib_mbb_sed_weight(β, Tdust, ν0, ν; T_CMB=T_CMB)

Calculates the spectral energy distribution (SED) weight for a modified blackbody (MBB)
model, typically used for the Cosmic Infrared Background (CIB).

The weight is proportional to `ν^β * B(ν, T_dust) / (dB/dT)(ν, T_CMB)`.

# Arguments
- `β`: Power-law spectral index for the dust emissivity.
- `Tdust`: Dust temperature in Kelvin.
- `ν0`: Reference frequency for normalization.
- `ν`: Frequency at which to evaluate the SED.

# Keywords
- `T_CMB`: Temperature of the CMB in Kelvin.

# Returns
- The dimensionless SED weight.
"""
function cib_mbb_sed_weight(β, Tdust, ν0, ν; T_CMB=T_CMB)
    βu, Tdu, ν0u, νu, Tcu = promote(β, Tdust, ν0, ν, T_CMB)
    r = νu / ν0u
    return r^βu * Bnu_ratio(νu, ν0u, Tdu) / dBdT_ratio(νu, ν0u, Tcu)
end

"""
    dust_tt_power_law(ℓs, A_pivot, α, β, ν1, ν2, Tdust, ν0; ℓ_pivot=80, T_CMB=T_CMB)

Computes a power-law model for the thermal dust auto-correlation (TT) power spectrum.

# Arguments
- `ℓs`: An `AbstractVector` of multipoles.
- `A_pivot`: Amplitude of the power spectrum at `ℓ_pivot`.
- `α`: Power-law index for the multipole dependence.
- `β`: Spectral index for the dust emissivity.
- `ν1`, `ν2`: Frequencies of the two channels being correlated.
- `Tdust`: Dust temperature in Kelvin.
- `ν0`: Reference frequency for the SED calculation.

# Keywords
- `ℓ_pivot`: Pivot multipole, default is 80.
- `T_CMB`: Temperature of the CMB in Kelvin.

# Returns
- An `AbstractVector` containing the dust power spectrum `Dℓ` at each `ℓ` in `ℓs`.
"""
function dust_tt_power_law(ℓs::AbstractVector, A_pivot, α, β, ν1, ν2, Tdust, ν0;
    ℓ_pivot=80, T_CMB=T_CMB)
    s1 = cib_mbb_sed_weight(β, Tdust, ν0, ν1; T_CMB=T_CMB)
    s2 = cib_mbb_sed_weight(β, Tdust, ν0, ν2; T_CMB=T_CMB)
    # Also rename A80 to A_pivot to reflect its general nature
    return (ℓs ./ ℓ_pivot) .^ (α + 2) .* (A_pivot * s1 * s2)
end

"""
    cib_clustered_power(ℓs, A_CIB, α, β, ν, z, Tdust, ν0_cib; ℓ_pivot=3000, T_CMB=T_CMB)

Computes the clustered Cosmic Infrared Background (CIB) power spectrum.

# Arguments
- `ℓs`: An `AbstractVector` of multipoles.
- `A_CIB`: Amplitude of the CIB power spectrum.
- `α`: Power-law index for the multipole dependence.
- `β`: Spectral index for the dust emissivity.
- `ν`: Frequency.
- `z`: Redshift.
- `Tdust`: Dust temperature in Kelvin.
- `ν0_cib`: Reference frequency for the CIB SED.

# Keywords
- `ℓ_pivot`: Pivot multipole, default is 3000.
- `T_CMB`: Temperature of the CMB in Kelvin.

# Returns
- An `AbstractVector` containing the CIB power spectrum `Dℓ`.
"""
function cib_clustered_power(ℓs::AbstractVector,
    A_CIB, α, β, ν, z,
    Tdust, ν0_cib; ℓ_pivot=3000, T_CMB=T_CMB)
    s = cib_mbb_sed_weight(β, Tdust, ν0_cib, ν; T_CMB=T_CMB)
    Z = abs(z)
    return @. (A_CIB * s * s * Z) * (ℓs / ℓ_pivot)^α
end

function cib_clustered_power(ℓs::AbstractVector,
    A_CIB, α, β, ν1, ν2, z1, z2,
    Tdust, ν0_cib; ℓ_pivot=3000, T_CMB=T_CMB)
    s1 = cib_mbb_sed_weight(β, Tdust, ν0_cib, ν1; T_CMB=T_CMB)
    s2 = cib_mbb_sed_weight(β, Tdust, ν0_cib, ν2; T_CMB=T_CMB)

    return @. (A_CIB * s1 * s2 * sqrt(z1 * z2)) * (ℓs / ℓ_pivot)^α
end

"""
    tsz_cross_power(template, A_tSZ, ν1, ν2, ν0; T_CMB=T_CMB)

Computes the thermal Sunyaev-Zel'dovich (tSZ) cross-power spectrum by scaling a template.

# Arguments
- `template`: An `AbstractVector` representing the tSZ power spectrum shape `Dℓ` at a reference frequency.
- `A_tSZ`: Amplitude of the tSZ power spectrum.
- `ν1`, `ν2`: Frequencies of the two channels being correlated.
- `ν0`: Reference frequency for the tSZ spectral function.

# Keywords
- `T_CMB`: Temperature of the CMB in Kelvin.

# Returns
- An `AbstractVector` containing the tSZ cross-power spectrum `Dℓ`.
"""
function tsz_cross_power(template::AbstractVector, A_tSZ, ν1, ν2, ν0; T_CMB=T_CMB)
    # Preserve AD types (Dual, BigFloat, etc.)
    A, ν1_, ν2_, ν0_, T_CMB_ = promote(A_tSZ, ν1, ν2, ν0, T_CMB)

    s1 = tsz_g_ratio(ν1_, ν0_, T_CMB_)
    s2 = tsz_g_ratio(ν2_, ν0_, T_CMB_)

    # Single-pass broadcast; result eltype promotes with A,s1,s2
    return @. template * (A * s1 * s2)
end

"""
    tsz_cib_cross_power(ℓs, ξ, A_tSZ, A_CIB, α, β, z1, z2, ν_cib1, ν_cib2, ν_tsz1, ν_tsz2, tsz_template, ν0_tsz, Tdust, ν0_cib; ℓ_pivot=3000, T_CMB=T_CMB)

Computes the cross-correlation power spectrum between the tSZ effect and the CIB.

# Arguments
- `ℓs`: Vector of multipoles.
- `ξ`: tSZ-CIB correlation coefficient.
- `A_tSZ`, `A_CIB`: Amplitudes for tSZ and CIB power spectra.
- `α`, `β`: Power-law indices for CIB.
- `z1`, `z2`: Redshifts for CIB channels.
- `ν_cib1`, `ν_cib2`: Frequencies for CIB channels.
- `ν_tsz1`, `ν_tsz2`: Frequencies for tSZ channels.
- `tsz_template`: Power spectrum template for tSZ.
- `ν0_tsz`, `ν0_cib`: Reference frequencies for tSZ and CIB.
- `Tdust`: Dust temperature.

# Keywords
- `ℓ_pivot`: Pivot multipole for CIB, default is 3000.
- `T_CMB`: Temperature of the CMB.

# Returns
- An `AbstractVector` of the tSZ-CIB cross-power spectrum `Dℓ`.
"""
function tsz_cib_cross_power(
    ℓs::AbstractVector,
    ξ, A_tSZ, A_CIB, α, β, z1, z2,
    ν_cib1, ν_cib2, ν_tsz1, ν_tsz2,
    tsz_template::AbstractVector,
    ν0_tsz, Tdust, ν0_cib; ℓ_pivot=3000, T_CMB=T_CMB
)
    @assert length(ℓs) == length(tsz_template)

    # CIB autos
    cib_11 = cib_clustered_power(ℓs, A_CIB, α, β, ν_cib1, ν_cib1, z1, z1, Tdust, ν0_cib; ℓ_pivot=ℓ_pivot, T_CMB=T_CMB)
    cib_22 = cib_clustered_power(ℓs, A_CIB, α, β, ν_cib2, ν_cib2, z2, z2, Tdust, ν0_cib; ℓ_pivot=ℓ_pivot, T_CMB=T_CMB)

    # tSZ autos (ν0 as keyword to match loaded method)
    tsz_11 = tsz_cross_power(tsz_template, A_tSZ, ν_tsz1, ν_tsz1, ν0_tsz; T_CMB=T_CMB)
    tsz_22 = tsz_cross_power(tsz_template, A_tSZ, ν_tsz2, ν_tsz2, ν0_tsz; T_CMB=T_CMB)

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

Calculates the derivative `dCℓ/dℓ` from the power spectrum `Dℓ`.

Note that `Dℓ = ℓ(ℓ+1)/(2π) * Cℓ`. The derivative is computed using central
differences for interior points and forward/backward differences at the boundaries.

# Arguments
- `ℓs`: An `AbstractVector` of multipoles.
- `Dℓ`: An `AbstractVector` of power spectrum values `Dℓ`.

# Returns
- An `AbstractVector` representing the derivative `dCℓ/dℓ`.
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

Calculates the change in the power spectrum `Dℓ` due to the super-sample lensing (SSL) effect.

# Arguments
- `ℓs`: An `AbstractVector` of multipoles.
- `κ`: The convergence field value.
- `Dℓ`: The unperturbed power spectrum `Dℓ`.

# Returns
- An `AbstractVector` representing the change in `Dℓ` from the SSL effect.
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

Calculates the change in the power spectrum `Dℓ` due to relativistic aberration.

# Arguments
- `ℓs`: An `AbstractVector` of multipoles.
- `ab_coeff`: The aberration coefficient (related to observer velocity).
- `Dℓ`: The unperturbed power spectrum `Dℓ`.

# Returns
- An `AbstractVector` representing the change in `Dℓ` from aberration.
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

Computes a shot noise power spectrum, which scales as `ℓ²`.

The model is `Dℓ = A_ℓ0 * (ℓ/ℓ0)²`.

# Arguments
- `ℓs`: An `AbstractVector` of multipoles.
- `A_ℓ0`: The amplitude of the shot noise `Dℓ` at `ℓ0`.

# Keywords
- `ℓ0`: The reference multipole, default is 3000.

# Returns
- An `AbstractVector` of the shot noise power `Dℓ`.
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

Calculates the Gaussian beam window function `B(ℓ)` for a given FWHM.

The window function is given by `B(ℓ) = exp(-0.5 * ℓ * (ℓ + 1) * σ²)`.

# Arguments
- `fwhm_arcmin`: The beam FWHM in arcminutes.
- `ells`: An `AbstractVector` of multipoles.

# Returns
- An `AbstractVector` containing the beam window function values.
"""
function gaussian_beam_window(fwhm_arcmin, ells::AbstractVector)
    σ = fwhm_arcmin_to_sigma_rad(fwhm_arcmin)
    return @. exp(-0.5 * ells * (ells + 1) * σ^2)
end
