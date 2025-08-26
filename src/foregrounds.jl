function planck_common_vars(ν, ν0, T)
    νu, ν0u, Tu = promote(ν, ν0, T)
    r = νu / ν0u
    x0 = Ghz_Kelvin * ν0u / Tu
    x = r * x0
    return r, x, x0
end

function planck_bnu_ratio(ν, ν0, T)
    r, x, x0 = planck_common_vars(ν, ν0, T)
    return r^3 * expm1(x0) / expm1(x)
end

function planck_dBdT_ratio(ν, ν0, T)
    r, x, x0 = planck_common_vars(ν, ν0, T)

    # use exp(x)/(exp(x)-1)^2 = 1/(4*sinh(x/2)^2)
    s0 = sinh(x0 / 2)
    s = sinh(x / 2)

    return r^4 * (s0 * s0) / (s * s)
end

function tsz_scaling(ν, ν0, T)
    r, x, x0 = planck_common_vars(ν, ν0, T)
    g0 = x0 * (one(x0) + 2 / expm1(x0)) - 4
    g = x * (one(x) + 2 / expm1(x)) - 4
    return g / g0
end

function cib_sed_factor(β, Tdust, ν0, ν; TCMB=T_CMB)
    βu, Tdu, ν0u, νu, Tcu = promote(β, Tdust, ν0, ν, TCMB)
    r = νu / ν0u
    return r^βu * planck_bnu_ratio(νu, ν0u, Tdu) / planck_dBdT_ratio(νu, ν0u, Tcu)
end

function galactic_dust_power(ℓs::AbstractVector, A_pivot, α, β, ν1, ν2, Tdust, ν0;
    ℓ_pivot=80, TCMB=T_CMB)
    s1 = cib_sed_factor(β, Tdust, ν0, ν1; TCMB=TCMB)
    s2 = cib_sed_factor(β, Tdust, ν0, ν2; TCMB=TCMB)
    # Also rename A80 to A_pivot to reflect its general nature
    return (ℓs ./ ℓ_pivot) .^ (α + 2) .* (A_pivot * s1 * s2)
end

function cib_cluster_auto(ℓs::AbstractVector,
    A_CIB, α, β, ν, z,
    Tdust, ν0_cib; ℓ_pivot=3000, TCMB=T_CMB)
    s = cib_sed_factor(β, Tdust, ν0_cib, ν; TCMB=TCMB)
    Z = abs(z)
    return @. (A_CIB * s * s * Z) * (ℓs / ℓ_pivot)^α
end

function tsz_power(template::AbstractVector, A_tSZ, ν1, ν2, ν0; T=T_CMB)
    # Preserve AD types (Dual, BigFloat, etc.)
    A, ν1_, ν2_, ν0_, T_ = promote(A_tSZ, ν1, ν2, ν0, T)

    s1 = tsz_scaling(ν1_, ν0_, T_)
    s2 = tsz_scaling(ν2_, ν0_, T_)

    # Single-pass broadcast; result eltype promotes with A,s1,s2
    return @. template * (A * s1 * s2)
end

function tsz_cib_correlation(
    ℓs::AbstractVector,
    ξ, A_tSZ, A_CIB, α, β, z1, z2,
    ν_cib1, ν_cib2, ν_tsz1, ν_tsz2,
    tsz_template::AbstractVector,
    ν0_tsz, Tdust, ν0_cib; ℓ_pivot=3000, TCMB=T_CMB
)
    @assert length(ℓs) == length(tsz_template)

    # CIB autos
    cib_11 = cib_cluster_auto(ℓs, A_CIB, α, β, ν_cib1, z1, Tdust, ν0_cib; ℓ_pivot=ℓ_pivot, TCMB=TCMB)
    cib_22 = cib_cluster_auto(ℓs, A_CIB, α, β, ν_cib2, z2, Tdust, ν0_cib)

    # tSZ autos (ν0 as keyword to match loaded method)
    tsz_11 = tsz_power(tsz_template, A_tSZ, ν_tsz1, ν_tsz1, ν0_tsz; T=TCMB)
    tsz_22 = tsz_power(tsz_template, A_tSZ, ν_tsz2, ν_tsz2, ν0_tsz; T=TCMB)

    return @. -ξ * (sqrt(abs(tsz_11 * cib_22)) + sqrt(abs(tsz_22 * cib_11)))
end


function tsz_scaling(ν, ν0, T)
    νu, ν0u, Tu = promote(ν, ν0, T)

    r = νu / ν0u
    x0 = Ghz_Kelvin * ν0u / Tu
    x = r * x0

    # Stable form: coth(x/2) = 1 + 2/(exp(x) - 1)
    g0 = x0 * (one(x0) + 2 / expm1(x0)) - 4
    g = x * (one(x) + 2 / expm1(x)) - 4

    return g / g0
end

function ksz_power(template::AbstractVector, A3000)
    # Broadcasted multiply promotes types automatically (e.g., Dual, BigFloat)
    return @. template * A3000
end

function cl_derivative_from_dl(ℓs::AbstractVector, Dℓ::AbstractVector)
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

function supersample_lensing(ℓs::AbstractVector, κ, Dℓ::AbstractVector)
    @assert length(ℓs) == length(Dℓ) "ells and Dℓ must have the same length"

    dCℓ = cl_derivative_from_dl(ℓs, Dℓ)

    out = similar(Dℓ)
    @inbounds @simd for i in eachindex(out, ℓs, dCℓ, Dℓ)
        ℓ = ℓs[i]
        pref = (ℓ * ℓ * (ℓ + 1)) / (2π)  # ℓ * ℓ(ℓ+1)/(2π)
        ssl = pref * dCℓ[i] + 2 * Dℓ[i]
        out[i] = -κ * ssl
    end
    return out
end

function aberration_correction(ℓs::AbstractVector, ab_coeff, Dℓ::AbstractVector)
    @assert length(ℓs) == length(Dℓ) "ells and Dℓ must have the same length"
    dCℓ = cl_derivative_from_dl(ℓs, Dℓ)

    out = similar(Dℓ)
    @inbounds @simd for i in eachindex(out, ℓs, dCℓ, Dℓ)
        ℓ = ℓs[i]
        pref = (ℓ * ℓ * (ℓ + 1)) / (2π)   # ℓ * ℓ(ℓ+1)/(2π)
        out[i] = -ab_coeff * dCℓ[i] * pref
    end
    return out
end

@inline function calibration(cal1, cal2, cal3, cal4)
    return (cal1 * cal2 + cal3 * cal4) / 2
end

function poisson_power(ℓs::AbstractVector, A_ℓ0; ℓ0=3000)
    s = A_ℓ0 / (ℓ0 * ℓ0)
    return @. ℓs * ℓs * s
end
