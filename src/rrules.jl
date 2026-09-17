"""
    rrules.jl

ChainRulesCore.rrule definitions for the cross-spectrum kernels in cross.jl.

These exist purely to speed up reverse-mode differentiation (Mooncake).
ForwardDiff does not use rrules — it goes through the original forward
implementations in cross.jl unchanged.

Each rrule:
  1. Computes the forward result by calling the existing kernel.
  2. Returns a pullback closure that maps the output cotangent back to
     input cotangents using BLAS for achromatic contractions and direct loops
     for chromatic contractions.

All incoming cotangents are `unthunk`ed defensively because Mooncake passes
`InplaceableThunk`s into pullbacks.

Survey-specific rrules (e.g. `theory_vector_core` which depends on
`ACTData`) live in their downstream packages and are *not* included here.
"""

using ChainRulesCore: rrule, NoTangent, ProjectTo, unthunk
using LinearAlgebra: dot, transpose

function ChainRulesCore.rrule(::typeof(window_convolution),
                              window::AbstractMatrix, spectrum::AbstractVector)
    projection = window_convolution(window, spectrum)
    project_spectrum = ProjectTo(spectrum)

    function window_convolution_pullback(projection̄_thunked)
        projection̄ = unthunk(projection̄_thunked)
        spectrum̄ = project_spectrum(window * projection̄)
        return NoTangent(), NoTangent(), spectrum̄
    end

    return projection, window_convolution_pullback
end

function ChainRulesCore.rrule(::typeof(_fixed_beam_product),
                              beam::AbstractMatrix, values::AbstractVector)
    output = beam * values
    project_values = ProjectTo(values)
    function fixed_beam_product_pullback(output̄_thunked)
        output̄ = unthunk(output̄_thunked)
        return NoTangent(), NoTangent(), project_values(transpose(beam) * output̄)
    end
    return output, fixed_beam_product_pullback
end

function ChainRulesCore.rrule(::typeof(_chromatic_ratio),
                              beam::AbstractMatrix, weights::AbstractVector,
                              sed_values::AbstractVector, denominator::AbstractVector)
    weighted_sed = weights .* sed_values
    numerator = beam * weighted_sed
    output = numerator ./ denominator
    project_weights = ProjectTo(weights)
    project_sed_values = ProjectTo(sed_values)
    project_denominator = ProjectTo(denominator)

    function chromatic_ratio_pullback(output̄_thunked)
        output̄ = unthunk(output̄_thunked)
        scaled_output̄ = output̄ ./ denominator
        weighted_sed̄ = transpose(beam) * scaled_output̄
        weights̄ = project_weights(sed_values .* weighted_sed̄)
        sed_values̄ = project_sed_values(weights .* weighted_sed̄)
        denominator̄ = project_denominator(-scaled_output̄ .* output)
        beam̄ = scaled_output̄ * transpose(weighted_sed)
        return NoTangent(), beam̄, weights̄, sed_values̄, denominator̄
    end

    return output, chromatic_ratio_pullback
end


function ChainRulesCore.rrule(::typeof(_fixed_chromatic_ratio),
                              beam::AbstractMatrix, weights::AbstractVector,
                              sed_values::AbstractVector, denominator::AbstractVector)
    weighted_sed = weights .* sed_values
    numerator = beam * weighted_sed
    output = numerator ./ denominator
    project_weights = ProjectTo(weights)
    project_sed_values = ProjectTo(sed_values)
    project_denominator = ProjectTo(denominator)

    function fixed_chromatic_ratio_pullback(output̄_thunked)
        output̄ = unthunk(output̄_thunked)
        scaled_output̄ = output̄ ./ denominator
        weighted_sed̄ = transpose(beam) * scaled_output̄
        weights̄ = project_weights(sed_values .* weighted_sed̄)
        sed_values̄ = project_sed_values(weights .* weighted_sed̄)
        denominator̄ = project_denominator(-scaled_output̄ .* output)
        return NoTangent(), NoTangent(), weights̄, sed_values̄, denominator̄
    end

    return output, fixed_chromatic_ratio_pullback
end

# ------------------------------------------------------------------ #
# factorized_cross(f, cl):  D[i,j,ℓ] = f[i] f[j] cl[ℓ]                  #
# ------------------------------------------------------------------ #

function ChainRulesCore.rrule(::typeof(factorized_cross),
                              f::AbstractVector{<:Real}, cl::AbstractVector{<:Real})
    n_freq = length(f)
    n_ell  = length(cl)
    D      = factorized_cross(f, cl)

    function factorized_cross_pullback(D̄_thunked)
        D̄      = unthunk(D̄_thunked)
        D̄_flat = reshape(D̄, n_freq * n_freq, n_ell)

        # M[i,j] = Σ_ℓ D̄[i,j,ℓ] cl[ℓ]
        M = reshape(D̄_flat * cl, n_freq, n_freq)

        # df̄[i] = Σ_j (M[i,j] + M[j,i]) f[j]
        df̄ = (M + transpose(M)) * f

        # dcl̄[ℓ] = Σ_{i,j} D̄[i,j,ℓ] f[i] f[j] = vec(f f')' * D̄[:,:,ℓ]
        dcl̄ = transpose(D̄_flat) * vec(f * transpose(f))

        return NoTangent(), df̄, dcl̄
    end

    return D, factorized_cross_pullback
end

# ------------------------------------------------------------------ #
# factorized_cross_te(fT, fE, cl):  D[i,j,ℓ] = fT[i] fE[j] cl[ℓ]        #
# ------------------------------------------------------------------ #

function ChainRulesCore.rrule(::typeof(factorized_cross_te),
                              fT::AbstractVector{<:Real},
                              fE::AbstractVector{<:Real},
                              cl::AbstractVector{<:Real})
    D      = factorized_cross_te(fT, fE, cl)
    n_freq = length(fT)
    n_ell  = length(cl)

    function factorized_cross_te_pullback(D̄_thunked)
        D̄      = unthunk(D̄_thunked)
        D̄_flat = reshape(D̄, n_freq * n_freq, n_ell)

        # M[i,j] = Σ_ℓ D̄[i,j,ℓ] cl[ℓ]
        M = reshape(D̄_flat * cl, n_freq, n_freq)

        # dfT̄[i] = Σ_j M[i,j] fE[j]
        dfT̄ = M * fE

        # dfĒ[j] = Σ_i M[i,j] fT[i]
        dfĒ = transpose(M) * fT

        # dcl̄[ℓ] = fT' * D̄[:,:,ℓ] * fE = vec(fT fE')' * D̄[:,:,ℓ]
        dcl̄ = transpose(D̄_flat) * vec(fT * transpose(fE))

        return NoTangent(), dfT̄, dfĒ, dcl̄
    end

    return D, factorized_cross_te_pullback
end

# ------------------------------------------------------------------ #
# factorized_cross(F::AbstractMatrix, cl::AbstractVector)             #
# D[i,j,ℓ] = F[i,ℓ] · F[j,ℓ] · cl[ℓ]                                 #
# ------------------------------------------------------------------ #

function ChainRulesCore.rrule(::typeof(factorized_cross),
                              F::AbstractMatrix{<:Real}, cl::AbstractVector{<:Real})
    D = factorized_cross(F, cl)
    n_freq, n_ell = size(F)
    project_F = ProjectTo(F)
    project_cl = ProjectTo(cl)

    function factorized_cross_matrix_pullback(D̄_thunked)
        D̄ = unthunk(D̄_thunked)
        T = promote_type(eltype(D̄), eltype(F), eltype(cl))
        dF̄  = zeros(T, size(F))
        dcl̄ = zeros(T, length(cl))

        @inbounds for ℓ in 1:n_ell
            cl_ℓ = cl[ℓ]
            for j in 1:n_freq, i in 1:n_freq
                weight = D̄[i, j, ℓ]
                dF̄[i, ℓ] += cl_ℓ * weight * F[j, ℓ]
                dF̄[j, ℓ] += cl_ℓ * weight * F[i, ℓ]
                dcl̄[ℓ] += weight * F[i, ℓ] * F[j, ℓ]
            end
        end

        return NoTangent(), project_F(dF̄), project_cl(dcl̄)
    end

    return D, factorized_cross_matrix_pullback
end

# ------------------------------------------------------------------ #
# factorized_cross_te(FT::AbstractMatrix, FE::AbstractMatrix, cl)     #
# D[i,j,ℓ] = FT[i,ℓ] · FE[j,ℓ] · cl[ℓ]                               #
# ------------------------------------------------------------------ #

function ChainRulesCore.rrule(::typeof(factorized_cross_te),
                              FT::AbstractMatrix{<:Real}, FE::AbstractMatrix{<:Real},
                              cl::AbstractVector{<:Real})
    D = factorized_cross_te(FT, FE, cl)
    n_freq, n_ell = size(FT)
    project_FT = ProjectTo(FT)
    project_FE = ProjectTo(FE)
    project_cl = ProjectTo(cl)

    function factorized_cross_te_matrix_pullback(D̄_thunked)
        D̄ = unthunk(D̄_thunked)

        T = promote_type(eltype(D̄), eltype(FT), eltype(FE), eltype(cl))
        dFT̄ = zeros(T, size(FT))
        dFĒ = zeros(T, size(FE))
        dcl̄ = zeros(T, length(cl))

        @inbounds for ℓ in 1:n_ell
            cl_ℓ = cl[ℓ]
            for j in 1:n_freq, i in 1:n_freq
                weight = D̄[i, j, ℓ]
                dFT̄[i, ℓ] += cl_ℓ * weight * FE[j, ℓ]
                dFĒ[j, ℓ] += cl_ℓ * weight * FT[i, ℓ]
                dcl̄[ℓ] += weight * FT[i, ℓ] * FE[j, ℓ]
            end
        end

        return NoTangent(), project_FT(dFT̄), project_FE(dFĒ), project_cl(dcl̄)
    end

    return D, factorized_cross_te_matrix_pullback
end

# ------------------------------------------------------------------ #
# correlated_cross(f, cl):  D[:,:,ℓ] = f' cl[:,:,ℓ] f                   #
# ------------------------------------------------------------------ #
# Forward (per ℓ):  D_ℓ = (f') · C_ℓ · f
# Reverse (per ℓ):
#   dC̄[:,:,ℓ] = f · D̄_ℓ · f'
#   df̄ contribution = C_ℓ · (f · D̄_ℓ') + C_ℓ' · (f · D̄_ℓ)
# ------------------------------------------------------------------ #

function ChainRulesCore.rrule(::typeof(correlated_cross),
                              f::AbstractMatrix{<:Real},
                              cl::AbstractArray{<:Real,3})
    D = correlated_cross(f, cl)
    n_comp, n_freq = size(f)
    n_ell = size(cl, 3)
    project_f = ProjectTo(f)
    project_cl = ProjectTo(cl)

    function correlated_cross_pullback(D̄_thunked)
        D̄  = unthunk(D̄_thunked)
        T  = promote_type(eltype(D̄), eltype(f), eltype(cl))
        df̄  = zeros(T, n_comp, n_freq)
        mixing = kron(transpose(f), transpose(f))
        cl_flat = reshape(cl, n_comp * n_comp, n_ell)
        D̄_flat = reshape(D̄, n_freq * n_freq, n_ell)
        dcl̄ = reshape(transpose(mixing) * D̄_flat, n_comp, n_comp, n_ell)
        dmixing = reshape(D̄_flat * transpose(cl_flat),
                          n_freq, n_freq, n_comp, n_comp)

        @inbounds for n in 1:n_comp, k in 1:n_comp,
                         j in 1:n_freq, i in 1:n_freq
            weight = dmixing[i, j, k, n]
            df̄[k, i] += weight * f[n, j]
            df̄[n, j] += weight * f[k, i]
        end

        return NoTangent(), project_f(df̄), project_cl(dcl̄)
    end

    return D, correlated_cross_pullback
end

# ------------------------------------------------------------------ #
# assemble_TT — fused TT-spectrum rrule                                #
# ------------------------------------------------------------------ #

function _factorized_term_grads(D̄_flat::AbstractMatrix, cl::AbstractVector,
                                f::AbstractVector, α::Real)
    n = length(f)
    M = reshape(D̄_flat * cl, n, n)
    d_α = dot(f, M, f)
    d_f = α * ((M + transpose(M)) * f)
    d_cl = α * (transpose(D̄_flat) * vec(f * transpose(f)))
    return d_α, d_f, d_cl
end

function ChainRulesCore.rrule(::typeof(assemble_TT),
                              a_p::Real, a_gtt::Real, a_s::Real,
                              f_ksz::AbstractVector{<:Real},   f_cibp::AbstractVector{<:Real},
                              f_dust::AbstractVector{<:Real},  f_radio::AbstractVector{<:Real},
                              f_tsz::AbstractVector{<:Real},   f_cibc::AbstractVector{<:Real},
                              cl_ksz::AbstractVector{<:Real},  cl_cibp::AbstractVector{<:Real},
                              cl_dustT::AbstractVector{<:Real}, cl_radio::AbstractVector{<:Real},
                              cl_tsz::AbstractVector{<:Real},  cl_cibc::AbstractVector{<:Real},
                              cl_szxcib::AbstractVector{<:Real})
    n_freq = length(f_ksz)
    n_ell  = length(cl_ksz)
    D = assemble_TT(a_p, a_gtt, a_s,
                    f_ksz, f_cibp, f_dust, f_radio, f_tsz, f_cibc,
                    cl_ksz, cl_cibp, cl_dustT, cl_radio,
                    cl_tsz, cl_cibc, cl_szxcib)

    function assemble_TT_pullback(D̄_thunked)
        D̄ = unthunk(D̄_thunked)
        D̄_flat = reshape(D̄, n_freq * n_freq, n_ell)

        # kSZ: cl_ksz is pre-scaled (α = 1 implicitly)
        M_ksz    = reshape(D̄_flat * cl_ksz, n_freq, n_freq)
        d_f_ksz  = (M_ksz + transpose(M_ksz)) * f_ksz
        d_cl_ksz = transpose(D̄_flat) * vec(f_ksz * transpose(f_ksz))

        # Standard factorized terms (α explicit)
        d_a_p,   d_f_cibp,  d_cl_cibp  = _factorized_term_grads(D̄_flat, cl_cibp,  f_cibp,  a_p)
        d_a_gtt, d_f_dust,  d_cl_dustT = _factorized_term_grads(D̄_flat, cl_dustT, f_dust,  a_gtt)
        d_a_s,   d_f_radio, d_cl_radio = _factorized_term_grads(D̄_flat, cl_radio, f_radio, a_s)

        # tSZ auto: f_tsz f_tsz' cl_tsz   (α = 1)
        M_tsz = reshape(D̄_flat * cl_tsz, n_freq, n_freq)
        d_f_tsz_auto = (M_tsz + transpose(M_tsz)) * f_tsz
        d_cl_tsz     = transpose(D̄_flat) * vec(f_tsz * transpose(f_tsz))

        # CIB-clustered auto: f_cibc f_cibc' cl_cibc   (α = 1)
        M_cibc = reshape(D̄_flat * cl_cibc, n_freq, n_freq)
        d_f_cibc_auto = (M_cibc + transpose(M_cibc)) * f_cibc
        d_cl_cibc     = transpose(D̄_flat) * vec(f_cibc * transpose(f_cibc))

        # tSZ × CIB cross: (f_tsz f_cibc' + f_cibc f_tsz') cl_szxcib   (α = 1)
        M_sxc = reshape(D̄_flat * cl_szxcib, n_freq, n_freq)
        d_f_tsz_cross  = M_sxc * f_cibc + transpose(M_sxc) * f_cibc
        d_f_cibc_cross = M_sxc * f_tsz  + transpose(M_sxc) * f_tsz
        d_cl_szxcib    = transpose(D̄_flat) *
                         vec(f_tsz * transpose(f_cibc) + f_cibc * transpose(f_tsz))

        d_f_tsz  = d_f_tsz_auto  + d_f_tsz_cross
        d_f_cibc = d_f_cibc_auto + d_f_cibc_cross

        return (NoTangent(),
                d_a_p, d_a_gtt, d_a_s,
                d_f_ksz, d_f_cibp, d_f_dust, d_f_radio,
                d_f_tsz, d_f_cibc,
                d_cl_ksz, d_cl_cibp, d_cl_dustT, d_cl_radio,
                d_cl_tsz, d_cl_cibc, d_cl_szxcib)
    end

    return D, assemble_TT_pullback
end

function ChainRulesCore.rrule(::typeof(assemble_TT),
                              a_p::Real, a_gtt::Real, a_s::Real,
                              f_ksz::AbstractMatrix{<:Real},   f_cibp::AbstractMatrix{<:Real},
                              f_dust::AbstractMatrix{<:Real},  f_radio::AbstractMatrix{<:Real},
                              f_tsz::AbstractMatrix{<:Real},   f_cibc::AbstractMatrix{<:Real},
                              cl_ksz::AbstractVector{<:Real},  cl_cibp::AbstractVector{<:Real},
                              cl_dustT::AbstractVector{<:Real}, cl_radio::AbstractVector{<:Real},
                              cl_tsz::AbstractVector{<:Real},  cl_cibc::AbstractVector{<:Real},
                              cl_szxcib::AbstractVector{<:Real})
    D = assemble_TT(a_p, a_gtt, a_s,
                    f_ksz, f_cibp, f_dust, f_radio, f_tsz, f_cibc,
                    cl_ksz, cl_cibp, cl_dustT, cl_radio,
                    cl_tsz, cl_cibc, cl_szxcib)
    n_freq, n_ell = size(f_ksz)
    project_f = map(ProjectTo, (f_ksz, f_cibp, f_dust, f_radio, f_tsz, f_cibc))
    project_cl = map(ProjectTo, (cl_ksz, cl_cibp, cl_dustT, cl_radio,
                                 cl_tsz, cl_cibc, cl_szxcib))

    function assemble_TT_matrix_pullback(D̄_thunked)
        D̄ = unthunk(D̄_thunked)
        T = promote_type(eltype(D̄), typeof(a_p), typeof(a_gtt), typeof(a_s),
                         eltype(f_ksz), eltype(cl_ksz))
        d_f_ksz = zeros(T, size(f_ksz)); d_f_cibp = zeros(T, size(f_cibp))
        d_f_dust = zeros(T, size(f_dust)); d_f_radio = zeros(T, size(f_radio))
        d_f_tsz = zeros(T, size(f_tsz)); d_f_cibc = zeros(T, size(f_cibc))
        d_cl_ksz = zeros(T, n_ell); d_cl_cibp = zeros(T, n_ell)
        d_cl_dustT = zeros(T, n_ell); d_cl_radio = zeros(T, n_ell)
        d_cl_tsz = zeros(T, n_ell); d_cl_cibc = zeros(T, n_ell)
        d_cl_szxcib = zeros(T, n_ell)
        d_a_p = zero(T); d_a_gtt = zero(T); d_a_s = zero(T)

        @inbounds for ℓ in 1:n_ell
            cksz, ccp, cdt, crd = cl_ksz[ℓ], cl_cibp[ℓ], cl_dustT[ℓ], cl_radio[ℓ]
            ctsz, ccc, csxc = cl_tsz[ℓ], cl_cibc[ℓ], cl_szxcib[ℓ]
            for j in 1:n_freq, i in 1:n_freq
                w = D̄[i, j, ℓ]
                fki, fkj = f_ksz[i, ℓ], f_ksz[j, ℓ]
                fpi, fpj = f_cibp[i, ℓ], f_cibp[j, ℓ]
                fdi, fdj = f_dust[i, ℓ], f_dust[j, ℓ]
                fri, frj = f_radio[i, ℓ], f_radio[j, ℓ]
                fti, ftj = f_tsz[i, ℓ], f_tsz[j, ℓ]
                fci, fcj = f_cibc[i, ℓ], f_cibc[j, ℓ]

                d_f_ksz[i, ℓ] += w * fkj * cksz
                d_f_ksz[j, ℓ] += w * fki * cksz
                d_cl_ksz[ℓ] += w * fki * fkj

                d_f_tsz[i, ℓ] += w * (ftj * ctsz + fcj * csxc)
                d_f_tsz[j, ℓ] += w * (fti * ctsz + fci * csxc)
                d_f_cibc[i, ℓ] += w * (fcj * ccc + ftj * csxc)
                d_f_cibc[j, ℓ] += w * (fci * ccc + fti * csxc)
                d_cl_tsz[ℓ] += w * fti * ftj
                d_cl_cibc[ℓ] += w * fci * fcj
                d_cl_szxcib[ℓ] += w * (fti * fcj + fci * ftj)

                d_a_p += w * fpi * fpj * ccp
                d_f_cibp[i, ℓ] += w * a_p * fpj * ccp
                d_f_cibp[j, ℓ] += w * a_p * fpi * ccp
                d_cl_cibp[ℓ] += w * a_p * fpi * fpj

                d_a_gtt += w * fdi * fdj * cdt
                d_f_dust[i, ℓ] += w * a_gtt * fdj * cdt
                d_f_dust[j, ℓ] += w * a_gtt * fdi * cdt
                d_cl_dustT[ℓ] += w * a_gtt * fdi * fdj

                d_a_s += w * fri * frj * crd
                d_f_radio[i, ℓ] += w * a_s * frj * crd
                d_f_radio[j, ℓ] += w * a_s * fri * crd
                d_cl_radio[ℓ] += w * a_s * fri * frj
            end
        end

        return (NoTangent(), d_a_p, d_a_gtt, d_a_s,
                project_f[1](d_f_ksz), project_f[2](d_f_cibp),
                project_f[3](d_f_dust), project_f[4](d_f_radio),
                project_f[5](d_f_tsz), project_f[6](d_f_cibc),
                project_cl[1](d_cl_ksz), project_cl[2](d_cl_cibp),
                project_cl[3](d_cl_dustT), project_cl[4](d_cl_radio),
                project_cl[5](d_cl_tsz), project_cl[6](d_cl_cibc),
                project_cl[7](d_cl_szxcib))
    end

    return D, assemble_TT_matrix_pullback
end

# ------------------------------------------------------------------ #
# assemble_EE — fused EE-spectrum rrule                                #
# ------------------------------------------------------------------ #

function ChainRulesCore.rrule(::typeof(assemble_EE),
                              a_psee::Real, a_gee::Real,
                              f_radio_P::AbstractVector{<:Real},
                              f_dust_P::AbstractVector{<:Real},
                              cl_radio::AbstractVector{<:Real},
                              cl_dustE::AbstractVector{<:Real})
    n_freq = length(f_radio_P)
    n_ell  = length(cl_radio)
    D = assemble_EE(a_psee, a_gee, f_radio_P, f_dust_P, cl_radio, cl_dustE)

    function assemble_EE_pullback(D̄_thunked)
        D̄ = unthunk(D̄_thunked)
        D̄_flat = reshape(D̄, n_freq * n_freq, n_ell)

        d_a_psee, d_f_radio_P, d_cl_radio =
            _factorized_term_grads(D̄_flat, cl_radio, f_radio_P, a_psee)
        d_a_gee,  d_f_dust_P,  d_cl_dustE =
            _factorized_term_grads(D̄_flat, cl_dustE, f_dust_P,  a_gee)

        return (NoTangent(),
                d_a_psee, d_a_gee,
                d_f_radio_P, d_f_dust_P,
                d_cl_radio,  d_cl_dustE)
    end

    return D, assemble_EE_pullback
end

function ChainRulesCore.rrule(::typeof(assemble_EE),
                              a_psee::Real, a_gee::Real,
                              f_radio_P::AbstractMatrix{<:Real},
                              f_dust_P::AbstractMatrix{<:Real},
                              cl_radio::AbstractVector{<:Real},
                              cl_dustE::AbstractVector{<:Real})
    D = assemble_EE(a_psee, a_gee, f_radio_P, f_dust_P, cl_radio, cl_dustE)
    n_freq, n_ell = size(f_radio_P)
    project_f_radio = ProjectTo(f_radio_P); project_f_dust = ProjectTo(f_dust_P)
    project_cl_radio = ProjectTo(cl_radio); project_cl_dust = ProjectTo(cl_dustE)

    function assemble_EE_matrix_pullback(D̄_thunked)
        D̄ = unthunk(D̄_thunked)
        T = promote_type(eltype(D̄), typeof(a_psee), typeof(a_gee),
                         eltype(f_radio_P), eltype(cl_radio))
        d_f_radio = zeros(T, size(f_radio_P)); d_f_dust = zeros(T, size(f_dust_P))
        d_cl_radio = zeros(T, n_ell); d_cl_dust = zeros(T, n_ell)
        d_a_psee = zero(T); d_a_gee = zero(T)
        @inbounds for ℓ in 1:n_ell, j in 1:n_freq, i in 1:n_freq
            w = D̄[i, j, ℓ]
            fri, frj = f_radio_P[i, ℓ], f_radio_P[j, ℓ]
            fdi, fdj = f_dust_P[i, ℓ], f_dust_P[j, ℓ]
            d_a_psee += w * fri * frj * cl_radio[ℓ]
            d_f_radio[i, ℓ] += w * a_psee * frj * cl_radio[ℓ]
            d_f_radio[j, ℓ] += w * a_psee * fri * cl_radio[ℓ]
            d_cl_radio[ℓ] += w * a_psee * fri * frj
            d_a_gee += w * fdi * fdj * cl_dustE[ℓ]
            d_f_dust[i, ℓ] += w * a_gee * fdj * cl_dustE[ℓ]
            d_f_dust[j, ℓ] += w * a_gee * fdi * cl_dustE[ℓ]
            d_cl_dust[ℓ] += w * a_gee * fdi * fdj
        end
        return (NoTangent(), d_a_psee, d_a_gee,
                project_f_radio(d_f_radio), project_f_dust(d_f_dust),
                project_cl_radio(d_cl_radio), project_cl_dust(d_cl_dust))
    end
    return D, assemble_EE_matrix_pullback
end

# ------------------------------------------------------------------ #
# assemble_TE — fused TE-spectrum rrule                                #
# ------------------------------------------------------------------ #

function _te_term_grads(D̄_flat::AbstractMatrix, cl::AbstractVector,
                        fT::AbstractVector, fE::AbstractVector, α::Real)
    n = length(fT)
    M = reshape(D̄_flat * cl, n, n)
    d_α  = dot(fT, M, fE)
    d_fT = α * (M           * fE)
    d_fE = α * (transpose(M) * fT)
    d_cl = α * (transpose(D̄_flat) * vec(fT * transpose(fE)))
    return d_α, d_fT, d_fE, d_cl
end

function ChainRulesCore.rrule(::typeof(assemble_TE),
                              a_pste::Real, a_gte::Real,
                              f_radio_T::AbstractVector{<:Real},
                              f_radio_P::AbstractVector{<:Real},
                              f_dust_T::AbstractVector{<:Real},
                              f_dust_P::AbstractVector{<:Real},
                              cl_radio::AbstractVector{<:Real},
                              cl_dustE::AbstractVector{<:Real})
    n_freq = length(f_radio_T)
    n_ell  = length(cl_radio)
    D = assemble_TE(a_pste, a_gte,
                    f_radio_T, f_radio_P, f_dust_T, f_dust_P,
                    cl_radio, cl_dustE)

    function assemble_TE_pullback(D̄_thunked)
        D̄ = unthunk(D̄_thunked)
        D̄_flat = reshape(D̄, n_freq * n_freq, n_ell)

        d_a_pste, d_f_radio_T, d_f_radio_P, d_cl_radio =
            _te_term_grads(D̄_flat, cl_radio, f_radio_T, f_radio_P, a_pste)
        d_a_gte,  d_f_dust_T,  d_f_dust_P,  d_cl_dustE =
            _te_term_grads(D̄_flat, cl_dustE, f_dust_T,  f_dust_P,  a_gte)

        return (NoTangent(),
                d_a_pste, d_a_gte,
                d_f_radio_T, d_f_radio_P,
                d_f_dust_T,  d_f_dust_P,
                d_cl_radio,  d_cl_dustE)
    end

    return D, assemble_TE_pullback
end

function ChainRulesCore.rrule(::typeof(assemble_TE),
                              a_pste::Real, a_gte::Real,
                              f_radio_T::AbstractMatrix{<:Real},
                              f_radio_P::AbstractMatrix{<:Real},
                              f_dust_T::AbstractMatrix{<:Real},
                              f_dust_P::AbstractMatrix{<:Real},
                              cl_radio::AbstractVector{<:Real},
                              cl_dustE::AbstractVector{<:Real})
    D = assemble_TE(a_pste, a_gte, f_radio_T, f_radio_P,
                    f_dust_T, f_dust_P, cl_radio, cl_dustE)
    n_freq, n_ell = size(f_radio_T)
    project_f = map(ProjectTo, (f_radio_T, f_radio_P, f_dust_T, f_dust_P))
    project_cl_radio = ProjectTo(cl_radio); project_cl_dust = ProjectTo(cl_dustE)

    function assemble_TE_matrix_pullback(D̄_thunked)
        D̄ = unthunk(D̄_thunked)
        T = promote_type(eltype(D̄), typeof(a_pste), typeof(a_gte),
                         eltype(f_radio_T), eltype(cl_radio))
        d_f_radio_T = zeros(T, size(f_radio_T)); d_f_radio_P = zeros(T, size(f_radio_P))
        d_f_dust_T = zeros(T, size(f_dust_T)); d_f_dust_P = zeros(T, size(f_dust_P))
        d_cl_radio = zeros(T, n_ell); d_cl_dust = zeros(T, n_ell)
        d_a_pste = zero(T); d_a_gte = zero(T)
        @inbounds for ℓ in 1:n_ell, j in 1:n_freq, i in 1:n_freq
            w = D̄[i, j, ℓ]
            frti, frpj = f_radio_T[i, ℓ], f_radio_P[j, ℓ]
            fdti, fdpj = f_dust_T[i, ℓ], f_dust_P[j, ℓ]
            d_a_pste += w * frti * frpj * cl_radio[ℓ]
            d_f_radio_T[i, ℓ] += w * a_pste * frpj * cl_radio[ℓ]
            d_f_radio_P[j, ℓ] += w * a_pste * frti * cl_radio[ℓ]
            d_cl_radio[ℓ] += w * a_pste * frti * frpj
            d_a_gte += w * fdti * fdpj * cl_dustE[ℓ]
            d_f_dust_T[i, ℓ] += w * a_gte * fdpj * cl_dustE[ℓ]
            d_f_dust_P[j, ℓ] += w * a_gte * fdti * cl_dustE[ℓ]
            d_cl_dust[ℓ] += w * a_gte * fdti * fdpj
        end
        return (NoTangent(), d_a_pste, d_a_gte,
                project_f[1](d_f_radio_T), project_f[2](d_f_radio_P),
                project_f[3](d_f_dust_T), project_f[4](d_f_dust_P),
                project_cl_radio(d_cl_radio), project_cl_dust(d_cl_dust))
    end
    return D, assemble_TE_matrix_pullback
end
