"""
    cross.jl

Cross-spectrum assembly: outer-product of frequency SEDs and ℓ-templates.
Mirrors fgspectra/cross.py.

All functions operate on real-valued spectra and return 3D arrays of shape
(n_freq, n_freq, n_ell).
All are pure functions — no mutation — compatible with ForwardDiff and Mooncake.
"""

# ------------------------------------------------------------------ #
# Factorized cross-spectrum                                            #
# D[i,j,ℓ] = f[i] · f[j] · Cl[ℓ]                                     #
# ------------------------------------------------------------------ #

@inline function _require_lengths(expected::Integer, description::AbstractString,
                                  arrays...)
    all(array -> length(array) == expected, arrays) ||
        throw(DimensionMismatch("all $description must have length $expected"))
    return nothing
end

"""
    factorized_cross(f, cl)

Factorized cross-spectrum: outer product of SED vector with itself,
times a scalar template:

    D[i, j, ℓ] = f[i] · f[j] · Cl[ℓ]

Arguments:
- `f`:  SED values, shape (n_freq,)
- `cl`: D_ℓ template, shape (n_ell,)

Returns array of shape (n_freq, n_freq, n_ell).
"""
function factorized_cross(f::AbstractVector{<:Real}, cl::AbstractVector{<:Real})
    Base.require_one_based_indexing(f, cl)
    n_freq = length(f)
    n_ell  = length(cl)
    # outer[i,j] = f[i]*f[j], then broadcast with cl
    outer = f .* f'                                        # (n_freq, n_freq)
    return reshape(outer, n_freq, n_freq, 1) .* reshape(cl, 1, 1, n_ell)
end

"""
    factorized_cross(F::AbstractMatrix, cl::AbstractVector)

Chromatic factorized cross-spectrum: outer product of frequency- and ℓ-dependent
weights with an ℓ-template:

    D[i, j, ℓ] = F[i, ℓ] · F[j, ℓ] · Cl[ℓ]

Arguments:
- `F`:  chromatic SED weights, shape (n_freq, n_ell)
- `cl`: D_ℓ template, shape (n_ell,)

Returns array of shape (n_freq, n_freq, n_ell).
"""
function factorized_cross(F::AbstractMatrix{<:Real}, cl::AbstractVector{<:Real})
    Base.require_one_based_indexing(F, cl)
    n_freq, n_ell = size(F)
    length(cl) == n_ell ||
        throw(DimensionMismatch("cl length must match the second dimension of F"))
    F_i = reshape(F, n_freq, 1, n_ell)
    F_j = reshape(F, 1, n_freq, n_ell)
    cl_3d = reshape(cl, 1, 1, n_ell)
    return @. F_i * F_j * cl_3d
end

# ------------------------------------------------------------------ #
# TE factorized cross-spectrum                                         #
# D[i,j,ℓ] = fT[i] · fE[j] · Cl[ℓ]                                   #
# (T and E can have different SEDs, e.g. different beam normalizations) #
# ------------------------------------------------------------------ #

"""
    factorized_cross_te(fT, fE, cl)

TE cross-spectrum: product of separate T and E SED vectors:

    D[i, j, ℓ] = fT[i] · fE[j] · Cl[ℓ]

Arguments:
- `fT`: temperature SED values, shape (n_freq,)
- `fE`: E-mode SED values, shape (n_freq,)
- `cl`: D_ℓ template, shape (n_ell,)

Returns array of shape (n_freq, n_freq, n_ell).
"""
function factorized_cross_te(fT::AbstractVector{<:Real}, fE::AbstractVector{<:Real},
                             cl::AbstractVector{<:Real})
    Base.require_one_based_indexing(fT, fE, cl)
    n_freq = length(fT)
    n_ell  = length(cl)
    length(fE) == n_freq ||
        throw(DimensionMismatch("fT and fE must have the same length"))
    outer  = fT .* fE'                                     # (n_freq, n_freq)
    return reshape(outer, n_freq, n_freq, 1) .* reshape(cl, 1, 1, n_ell)
end

"""
    factorized_cross_te(FT::AbstractMatrix, FE::AbstractMatrix, cl::AbstractVector)

Chromatic TE cross-spectrum with separate T and E frequency- and ℓ-dependent weights:

    D[i, j, ℓ] = FT[i, ℓ] · FE[j, ℓ] · Cl[ℓ]

Arguments:
- `FT`: temperature chromatic SED weights, shape (n_freq, n_ell)
- `FE`: E-mode chromatic SED weights, shape (n_freq, n_ell)
- `cl`: D_ℓ template, shape (n_ell,)

Returns array of shape (n_freq, n_freq, n_ell).
"""
function factorized_cross_te(FT::AbstractMatrix{<:Real}, FE::AbstractMatrix{<:Real},
                             cl::AbstractVector{<:Real})
    Base.require_one_based_indexing(FT, FE, cl)
    n_freq, n_ell = size(FT)
    size(FE) == (n_freq, n_ell) ||
        throw(DimensionMismatch("FE shape must match FT shape"))
    length(cl) == n_ell ||
        throw(DimensionMismatch("cl length must match the second dimension of FT"))
    FT_i = reshape(FT, n_freq, 1, n_ell)
    FE_j = reshape(FE, 1, n_freq, n_ell)
    cl_3d = reshape(cl, 1, 1, n_ell)
    return @. FT_i * FE_j * cl_3d
end

# ------------------------------------------------------------------ #
# Correlated cross-spectrum (tSZ + CIB + cross)                        #
# D[i,j,ℓ] = Σ_{k,n} f[k,i] · f[n,j] · C[k,n,ℓ]                     #
# ------------------------------------------------------------------ #

"""
    correlated_cross(f, cl)

Correlated cross-spectrum for multiple components:

    D[i, j, ℓ] = Σ_{k,n} f[k, i] · f[n, j] · C[k, n, ℓ]

This is the ACT DR6 model for tSZ + CIB clustered + tSZ×CIB:
- k=1: ThermalSZ;  k=2: CIB (MBB)
- C[1,1]: tSZ template × a_tSZ  (with tilt)
- C[2,2]: CIB clustered template × a_c
- C[1,2] = C[2,1]: cross template × (−ξ√(a_tSZ · a_c))

Arguments:
- `f`:  SED matrix, shape (n_comp, n_freq)
- `cl`: covariance spectrum tensor, shape (n_comp, n_comp, n_ell)

Returns array of shape (n_freq, n_freq, n_ell).
Matches `CorrelatedFactorizedCrossSpectrum` in fgspectra/cross.py.
"""
function correlated_cross(f::AbstractMatrix{<:Real}, cl::AbstractArray{<:Real,3})
    Base.require_one_based_indexing(f, cl)
    n_comp, n_freq = size(f)
    size(cl, 1) == n_comp && size(cl, 2) == n_comp ||
        throw(DimensionMismatch("cl component dimensions must match the first dimension of f"))
    n_ell          = size(cl, 3)
    mixing = kron(transpose(f), transpose(f))
    result = mixing * reshape(cl, n_comp * n_comp, n_ell)
    return reshape(result, n_freq, n_freq, n_ell)
end

# ------------------------------------------------------------------ #
# Helper: build the 2×2×n_ell covariance spectrum for tSZ+CIB          #
# ------------------------------------------------------------------ #

"""
    build_szxcib_cl(cl_tsz, cl_cibc, cl_cross)

Assemble the 2×2 component-component D_ℓ covariance matrix for the
tSZ+CIB correlated model.

Layout:
    C[1,1,ℓ] = cl_tsz[ℓ]    (tSZ auto, with amplitude a_tSZ and tilt)
    C[2,2,ℓ] = cl_cibc[ℓ]   (CIB clustered auto, amplitude a_c)
    C[1,2,ℓ] = C[2,1,ℓ] = cl_cross[ℓ]  (−ξ√(a_tSZ·a_c) × template)

All inputs are Vectors of length n_ell (already amplitude-multiplied).
"""
function build_szxcib_cl(cl_tsz::AbstractVector{<:Real},
                         cl_cibc::AbstractVector{<:Real},
                         cl_cross::AbstractVector{<:Real})
    Base.require_one_based_indexing(cl_tsz, cl_cibc, cl_cross)
    n_ell = length(cl_tsz)
    _require_lengths(n_ell, "component spectra", cl_cibc, cl_cross)
    # Pure construction — no mutation — compatible with ForwardDiff and Mooncake
    layer11 = reshape(cl_tsz,  1, 1, n_ell)
    layer12 = reshape(cl_cross, 1, 1, n_ell)
    layer21 = reshape(cl_cross, 1, 1, n_ell)
    layer22 = reshape(cl_cibc,  1, 1, n_ell)
    row1 = cat(layer11, layer12; dims=2)   # (1, 2, n_ell)
    row2 = cat(layer21, layer22; dims=2)   # (1, 2, n_ell)
    return cat(row1, row2; dims=1)          # (2, 2, n_ell)
end

# ------------------------------------------------------------------ #
# Fused TT/EE/TE foreground assemblers                                #
# Each fuses the α .* factorized .+ correlated .+ ... chain into a    #
# single primitive so the Mooncake tape collapses to one entry per    #
# spectrum (replaces ~15k tape entries from per-element broadcasts).  #
# ------------------------------------------------------------------ #

"""
    assemble_TT(a_p, a_gtt, a_s,
                f_ksz, f_cibp, f_dust, f_radio, f_tsz, f_cibc,
                cl_ksz, cl_cibp, cl_dustT, cl_radio,
                cl_tsz, cl_cibc, cl_szxcib)

Fused TT foreground assembly — equivalent to:

    factorized_cross(f_ksz,  cl_ksz)             .+   # cl_ksz is already A_kSZ-scaled
    correlated_cross(vcat(f_tsz', f_cibc'), build_szxcib_cl(cl_tsz, cl_cibc, cl_szxcib)) .+
    a_p   .* factorized_cross(f_cibp, cl_cibp)  .+
    a_gtt .* factorized_cross(f_dust, cl_dustT) .+
    a_s   .* factorized_cross(f_radio, cl_radio)

`cl_ksz` is expected to be pre-scaled by `A_kSZ` (via `CMBForegrounds.ksz_template_scaled`).
Returns array of shape (n_freq, n_freq, n_ell).
"""
function assemble_TT(a_p::Real, a_gtt::Real, a_s::Real,
                     f_ksz::AbstractVector{<:Real},   f_cibp::AbstractVector{<:Real},
                     f_dust::AbstractVector{<:Real},  f_radio::AbstractVector{<:Real},
                     f_tsz::AbstractVector{<:Real},   f_cibc::AbstractVector{<:Real},
                     cl_ksz::AbstractVector{<:Real},  cl_cibp::AbstractVector{<:Real},
                     cl_dustT::AbstractVector{<:Real}, cl_radio::AbstractVector{<:Real},
                     cl_tsz::AbstractVector{<:Real},  cl_cibc::AbstractVector{<:Real},
                     cl_szxcib::AbstractVector{<:Real})
    Base.require_one_based_indexing(f_ksz, f_cibp, f_dust, f_radio,
                                    f_tsz, f_cibc, cl_ksz, cl_cibp,
                                    cl_dustT, cl_radio, cl_tsz, cl_cibc,
                                    cl_szxcib)
    n_freq = length(f_ksz)
    n_ell  = length(cl_ksz)
    _require_lengths(n_freq, "frequency vectors", f_cibp, f_dust, f_radio,
                     f_tsz, f_cibc)
    _require_lengths(n_ell, "angular spectra", cl_cibp, cl_dustT, cl_radio,
                     cl_tsz, cl_cibc, cl_szxcib)
    T = promote_type(typeof(a_p), typeof(a_gtt), typeof(a_s),
                     eltype(f_ksz), eltype(f_cibp), eltype(f_dust), eltype(f_radio),
                     eltype(f_tsz), eltype(f_cibc),
                     eltype(cl_ksz), eltype(cl_cibp), eltype(cl_dustT), eltype(cl_radio),
                     eltype(cl_tsz), eltype(cl_cibc), eltype(cl_szxcib))
    out = Array{T}(undef, n_freq, n_freq, n_ell)
    @inbounds for ℓ in 1:n_ell
        cksz, ccp, cdt, crd = cl_ksz[ℓ], cl_cibp[ℓ], cl_dustT[ℓ], cl_radio[ℓ]
        ctsz, ccc, csxc     = cl_tsz[ℓ], cl_cibc[ℓ], cl_szxcib[ℓ]
        for j in 1:n_freq
            ftj, fcj, fkj, fpj, fdj, frj =
                f_tsz[j], f_cibc[j], f_ksz[j], f_cibp[j], f_dust[j], f_radio[j]
            for i in 1:n_freq
                fti, fci, fki, fpi, fdi, fri =
                    f_tsz[i], f_cibc[i], f_ksz[i], f_cibp[i], f_dust[i], f_radio[i]
                out[i, j, ℓ] = fki * fkj * cksz +
                               fti  * ftj  * ctsz +
                               (fti * fcj + fci * ftj) * csxc +
                               fci  * fcj  * ccc +
                               a_p   * fpi * fpj * ccp +
                               a_gtt * fdi * fdj * cdt +
                               a_s   * fri * frj * crd
            end
        end
    end
    return out
end

"""
    assemble_EE(a_psee, a_gee, f_radio_P, f_dust_P, cl_radio, cl_dustE)

Fused EE foreground assembly — equivalent to:

    a_psee .* factorized_cross(f_radio_P, cl_radio) .+
    a_gee  .* factorized_cross(f_dust_P,  cl_dustE)
"""
function assemble_EE(a_psee::Real, a_gee::Real,
                     f_radio_P::AbstractVector{<:Real},
                     f_dust_P::AbstractVector{<:Real},
                     cl_radio::AbstractVector{<:Real},
                     cl_dustE::AbstractVector{<:Real})
    Base.require_one_based_indexing(f_radio_P, f_dust_P, cl_radio, cl_dustE)
    n_freq = length(f_radio_P)
    n_ell  = length(cl_radio)
    _require_lengths(n_freq, "frequency vectors", f_dust_P)
    _require_lengths(n_ell, "angular spectra", cl_dustE)
    T = promote_type(typeof(a_psee), typeof(a_gee),
                     eltype(f_radio_P), eltype(f_dust_P),
                     eltype(cl_radio), eltype(cl_dustE))
    out = Array{T}(undef, n_freq, n_freq, n_ell)
    @inbounds for ℓ in 1:n_ell
        crd, cdE = cl_radio[ℓ], cl_dustE[ℓ]
        for j in 1:n_freq
            frj, fdj = f_radio_P[j], f_dust_P[j]
            for i in 1:n_freq
                out[i, j, ℓ] = a_psee * f_radio_P[i] * frj * crd +
                               a_gee  * f_dust_P[i]  * fdj * cdE
            end
        end
    end
    return out
end

"""
    assemble_TE(a_pste, a_gte,
                f_radio_T, f_radio_P, f_dust_T, f_dust_P,
                cl_radio, cl_dustE)

Fused TE foreground assembly — equivalent to:

    a_pste .* factorized_cross_te(f_radio_T, f_radio_P, cl_radio) .+
    a_gte  .* factorized_cross_te(f_dust_T,  f_dust_P,  cl_dustE)
"""
function assemble_TE(a_pste::Real, a_gte::Real,
                     f_radio_T::AbstractVector{<:Real},
                     f_radio_P::AbstractVector{<:Real},
                     f_dust_T::AbstractVector{<:Real},
                     f_dust_P::AbstractVector{<:Real},
                     cl_radio::AbstractVector{<:Real},
                     cl_dustE::AbstractVector{<:Real})
    Base.require_one_based_indexing(f_radio_T, f_radio_P, f_dust_T,
                                    f_dust_P, cl_radio, cl_dustE)
    n_freq = length(f_radio_T)
    n_ell  = length(cl_radio)
    _require_lengths(n_freq, "frequency vectors", f_radio_P, f_dust_T, f_dust_P)
    _require_lengths(n_ell, "angular spectra", cl_dustE)
    T = promote_type(typeof(a_pste), typeof(a_gte),
                     eltype(f_radio_T), eltype(f_radio_P),
                     eltype(f_dust_T),  eltype(f_dust_P),
                     eltype(cl_radio),  eltype(cl_dustE))
    out = Array{T}(undef, n_freq, n_freq, n_ell)
    @inbounds for ℓ in 1:n_ell
        crd, cdE = cl_radio[ℓ], cl_dustE[ℓ]
        for j in 1:n_freq
            frPj, fdPj = f_radio_P[j], f_dust_P[j]
            for i in 1:n_freq
                out[i, j, ℓ] = a_pste * f_radio_T[i] * frPj * crd +
                               a_gte  * f_dust_T[i]  * fdPj * cdE
            end
        end
    end
    return out
end
