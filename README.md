# CMBForegrounds.jl

| **Documentation** | **Build Status** | **Code Coverage** | **Code Style** |
|:--------:|:----------------:|:----------------:|:----------------:|
| [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliacosmologicallikelihoods.github.io/CMBForegrounds.jl/dev) [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliacosmologicallikelihoods.github.io/CMBForegrounds.jl/stable) | [![Build status (Github Actions)](https://github.com/JuliaCosmologicalLikelihoods/CMBForegrounds.jl/workflows/CI/badge.svg)](https://github.com/JuliaCosmologicalLikelihoods/CMBForegrounds.jl/actions) | [![codecov](https://codecov.io/gh/JuliaCosmologicalLikelihoods/CMBForegrounds.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaCosmologicalLikelihoods/CMBForegrounds.jl) | [![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle) [![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac) |

`CMBForegrounds.jl` provides differentiable, survey-agnostic building blocks for CMB foreground power spectra and selected instrumental responses. Survey packages retain ownership of templates, passbands, parameter names and priors, data ordering, binning, and covariances.

The library dispatches on mathematical representations rather than survey names:

- angular shapes: power laws, exact Poisson spectra, templates, and tilted templates;
- SEDs: modified blackbody, radio, thermal SZ, and unit response;
- correlations: explicit cross-templates and geometric-mean prescriptions;
- passbands: effective frequencies and tabulated responses, including chromatic beams;
- instrumental operations: calibration, additive templates, T-to-E leakage, beam modes, super-sample lensing, and aberration.

Existing low-level foreground functions and fused `assemble_TT`, `assemble_TE`, and `assemble_EE` kernels remain available.

## Example

```julia
using CMBForegrounds

ells = collect(30:3000)

# D_ell = A (ell/ell_0)^alpha with a modified-blackbody SED.
dust = SkyComponent(
    ModifiedBlackbodySED(150.0, 19.6),
    PowerLawShape(500.0),
)
D_dust = eval_component(dust, ells, 90.0, 150.0, 8.0, 1.5; alpha=-0.6)

# Constant C_ell point sources: exact ell(ell+1) scaling in D_ell.
radio = SkyComponent(
    RadioSED(150.0; convention=:flux),
    PoissonShape(3000.0),
)
D_radio = eval_component(radio, ells, 90.0, 150.0, 3.0, -0.7)

# A dense template whose first sample corresponds to ell=0.
template = ones(3001)
tsz = SkyComponent(
    ThermalSZSED(143.0),
    TemplateShape(template; ell_0=3000, ell_min=0),
)
D_tsz = eval_component(tsz, ells, 90.0, 150.0, 4.0)
```

`PowerLawShape` and `PoissonShape` store the pivot only. Amplitudes and fitted slopes remain numerical arguments, so likelihood code controls parameter sharing without an additional parameter-management layer.

## Passbands and chromatic beams

```julia
nu = collect(range(130.0, 170.0; length=41))
transmission = @. exp(-0.5 * ((nu - 150.0) / 10.0)^2)
band = make_band(nu, transmission)

beam_matrix = [1 + 0.01 * (l / 3000) * (n - 150) / 20
               for l in ells, n in nu]
beam = ChromaticBeam(ells, beam_matrix)

sed = ModifiedBlackbodySED(150.0, 19.6)
weight_ell = sed_weight(sed, band, beam, 1.5)
```

`RawBand` and `shift_and_normalize` provide differentiable passband shifts. The chromatic beam is supplied as the already-evaluated matrix `beam[ell_index, nu_index]`; this package does not prescribe a survey beam model.

Pairwise `eval_component` calls accept scalar frequencies, `DeltaBand`s, and
tabulated `Band`s directly. A custom `AbstractSED` needs only a scalar
`sed_weight` method; ordinary and chromatic passband lifting is provided by the
package. Frequency grids must be finite and strictly increasing, and undefined
passband or chromatic normalizations raise an error.

## Correlated components

```julia
cross_shape = TemplateCorrelation(TemplateShape(template; ell_0=3000, ell_min=0))
f_tsz_90  = sed_weight(ThermalSZSED(143.0), 90.0)
f_tsz_150 = sed_weight(ThermalSZSED(143.0), 150.0)
f_cib_90  = sed_weight(ModifiedBlackbodySED(150.0, 25.0), 90.0, 1.75)
f_cib_150 = sed_weight(ModifiedBlackbodySED(150.0, 25.0), 150.0, 1.75)

D_cross = correlation_power(
    cross_shape, ells, 0.1, 4.0, 6.0,
    f_tsz_90, f_tsz_150, f_cib_90, f_cib_150,
)
```

`GeometricMeanCorrelation()` evaluates the alternative prescription from supplied auto-spectra or `SkyComponent`s. It is a distinct model and does not preserve the tSZ frequency sign across its null.

## Instrumental operations

```julia
D_calibrated = apply_calibration(D_dust, 1.01, 0.99; convention=:inverse)
D_with_template = add_template(D_calibrated, systematic_template, amplitude)

delta_TE = te_leakage(D_TT, gamma_E)
delta_EE = ee_leakage(D_TT, D_TE_ij, D_TE_ji, gamma_i, gamma_j)

D_ssl = apply_ssl(ells, kappa, D_cmb)
D_aberrated = apply_aberration(ells, aberration_coefficient, D_cmb)
```

Response provenance remains the caller's responsibility. Do not apply a correction twice when released spectra or covariances already include it.

## Conventions

- Angular evaluators return `D_ell`; `PowerLawShape` therefore takes the `D_ell` exponent directly.
- `PoissonShape` uses exact `ell(ell+1)` scaling, while legacy `shot_noise_power` intentionally implements an `ell^2` approximation.
- `TemplateShape` represents a dense integer multipole grid beginning at `ell_min`. `ell_0=nothing` means the input is already normalized.
- `TiltedTemplateShape(values, ell_0)` preserves the raw template normalization; wrap a pivot-normalized `TemplateShape` when `amp` denotes the physical pivot amplitude.
- `RadioSED(...; convention=:flux)` takes a flux-density index, while `convention=:rj` takes an RJ-temperature index. They obey `beta_rj = beta_flux - 2`.
- `:forward` calibration multiplies by gains; `:inverse` divides by them.
- Beam eigenmodes are fractional map-beam perturbations, and EE leakage takes the underlying unleaked ordered TE tensor.
- Fixed versus sampled parameters and all priors belong to the likelihood.

## Verification scope

Unit tests cover analytic identities, legacy-kernel parity, automatic-differentiation agreement, and synthetic compositions resembling common Planck/ACT/SPT modelling choices. They are **not** a substitute for likelihood-level comparisons against released survey model vectors. Such release-specific parity tests belong in the corresponding likelihood package together with its numerical assets.

## Development

```julia
julia --project=. -e 'using Pkg; Pkg.test()'
```

AD tests use `DifferentiationInterface` with ForwardDiff, Zygote, and Mooncake where supported.

## License

MIT
