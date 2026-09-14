# CMBForegrounds.jl

| **Documentation** | **Build Status** | **Code Coverage** | **Code Style** |
|:--------:|:----------------:|:----------------:|:----------------:|
| [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliacosmologicallikelihoods.github.io/CMBForegrounds.jl/dev) [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliacosmologicallikelihoods.github.io/CMBForegrounds.jl/stable) | [![Build status (Github Actions)](https://github.com/JuliaCosmologicalLikelihoods/CMBForegrounds.jl/workflows/CI/badge.svg)](https://github.com/JuliaCosmologicalLikelihoods/CMBForegrounds.jl/actions) | [![codecov](https://codecov.io/gh/JuliaCosmologicalLikelihoods/CMBForegrounds.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaCosmologicalLikelihoods/CMBForegrounds.jl) | [![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle) [![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac) |

`CMBForegrounds.jl` is a high-performance, differentiable Julia library for modeling astrophysical foregrounds, secondary anisotropies, and instrumental systematics in Cosmic Microwave Background (CMB) power spectra.

The package is **survey-agnostic by design**: rather than encoding survey names, it dispatches on foundational **mathematical representations** (angular shapes, spectral energy distributions, cross-correlations, bandpasses, and instrumental transformations). Empirical templates, measured bandpasses, beam transfer matrices, and likelihood covariances remain external user/likelihood assets.

---

## Key Features

- **Survey-Agnostic Multiple Dispatch**:
  - **Angular Shapes**: `PowerLawShape`, `PoissonShape`, `TemplateShape`, `TiltedTemplateShape`.
  - **Spectral Energy Distributions (SEDs)**: `ModifiedBlackbodySED`, `RadioSED`, `ThermalSZSED`, `ConstantSED`, `NoSED`.
  - **Component Composition**: `SkyComponent(shape, sed)` evaluated across effective frequencies or discretized bands.
  - **Cross-Correlations**: `TemplateCorrelation`, `GeometricMeanCorrelation`.
  - **Bandpass & Chromatic Beams**: `DeltaBand`, `Band`, `ChromaticBeam`, `integrate_chromatic_sed`, `eval_chromatic_sed_bands`.
  - **Instrumental Operations**: Map & spectrum calibrations (`:forward` and `:inverse`), additive templates, polarization leakage, beam eigenmodes, super-sample lensing (SSL), and relativistic aberration.
- **Ultra-High Performance**:
  - Zero-allocation inner loops and optimized matrix cross kernels (`factorized_cross`, `factorized_cross_te`).
  - Pre-allocated multi-channel fused assemblers (`assemble_TT`, `assemble_EE`, `assemble_TE`).
- **End-to-End Automatic Differentiation**:
  - Verified with **ForwardDiff**, **Mooncake**, and **Zygote** via `DifferentiationInterface`.
  - Custom `ChainRulesCore` rrules and dedicated Mooncake tape registrations in `CMBForegroundsMooncakeExt`.

---

## Quick Start

### 1. Sky Components (Angular Shape + SED)

```julia
using CMBForegrounds

# Define angular multipoles
ell = 2:3000

# 1. Thermal dust: power-law angular spectrum with modified blackbody SED
dust_shape = PowerLawShape(10.0, -0.6; ell_0=3000.0)
dust_sed = ModifiedBlackbodySED(1.53, 19.6; nu_0=353.0)
dust = SkyComponent(dust_shape, dust_sed)

# Evaluate cross-frequency power spectrum between 150 GHz and 220 GHz
cl_dust = eval_component(dust, 150.0, 220.0, ell)

# 2. Point sources: Poisson angular shape with power-law radio SED
radio_shape = PoissonShape(5.0; ell_0=3000.0)
radio_sed = RadioSED(-0.7; nu_0=150.0)
radio = SkyComponent(radio_shape, radio_sed)
cl_radio = eval_component(radio, 150.0, 150.0, ell)

# 3. Secondary anisotropy: Thermal Sunyaev-Zel'dovich with tabulated template
tsz_template = rand(3001) # e.g., from Shaw et al. or Battaglia et al.
tsz_shape = TemplateShape(tsz_template; ell_0=3000.0)
tsz_sed = ThermalSZSED(; nu_0=143.0)
tsz = SkyComponent(tsz_shape, tsz_sed)
cl_tsz = eval_component(tsz, 150.0, 150.0, ell)
```

### 2. Cross-Correlations

Correlate components (e.g. tSZ and CIB) using either template-based or geometric-mean representations:

```julia
# Template cross-correlation
tsz_cib_corr = TemplateCorrelation(tsz_cib_template; ell_0=3000.0, xi=-0.1)
cl_cross = correlation_power(tsz_cib_corr, ell)

# Geometric-mean cross-correlation: xi * sqrt(|C1 * C2|)
geom_corr = GeometricMeanCorrelation(dust_shape, radio_shape; xi=0.2)
cl_geom = correlation_power(geom_corr, ell)
```

### 3. Bandpasses and Chromatic Beams

Integrate SEDs across top-hat or measured transmission curves with optional chromatic beams:

```julia
nus = range(130.0, 170.0, length=40)
weights = exp.(-0.5 .* ((nus .- 150.0) ./ 10.0).^2)
band150 = Band(nus, weights)

# Chromatic beam: frequency-dependent FWHM
beam150 = ChromaticBeam(nus, 1.4 .* (150.0 ./ nus))

# Integrated SED weight for modified blackbody
w150 = integrate_chromatic_sed(dust_sed, band150; beam=beam150)
```

### 4. Composable Instrumental Operations

Apply calibration, polarization leakage, beam perturbations, or aberration to spectra:

```julia
# Map calibrations (forward: c1*c2; inverse: 1/(y1*y2))
cl_cal = apply_calibration(cl_dust, 1.01, 0.99; convention=:forward)

# Additive systematic templates (fixed or sampled amplitude)
cl_tot = add_template(cl_cal, template_array, 0.05)

# Polarization leakage from map-level response algebra
cl_te_obs = apply_te_leakage(cl_tt, cl_te, 0.005) # eta_leakage
cl_ee_obs = apply_ee_leakage(cl_tt, cl_te, cl_ee, 0.005)

# Relativistic aberration and super-sample lensing
cl_aberr = apply_aberration(cl_tot, dcl_dell, 0.00123)
cl_ssl   = apply_ssl(cl_tot, dcl_dell, 0.0005)
```

---

## Conventions & Physics Reference

| Quantity | Representation | Default Pivot | Notes |
|:---|:---|:---|:---|
| **Power-law slope** | `PowerLawShape(A, α)` | $\ell_0 = 3000$ | $D_\ell = A (\ell / \ell_0)^\alpha$. Note: legacy `dust_tt_power_law` took $\alpha + 2$. |
| **Poisson noise** | `PoissonShape(A)` | $\ell_0 = 3000$ | $D_\ell = A (\ell / \ell_0)^2$, equivalent to constant $C_\ell = A \times \frac{2\pi}{\ell_0^2}$. |
| **Radio index** | `RadioSED(α)` | $\nu_0 = 150\,\mathrm{GHz}$ | Rayleigh-Jeans temperature index $\alpha = \alpha_{\rm flux} - 2$. |
| **Modified Blackbody** | `ModifiedBlackbodySED(β, T)` | $\nu_0 = 353\,\mathrm{GHz}$ | Ratio of $\nu^\beta B_\nu(T)$ converted to thermodynamic $\Delta T_{\rm CMB}$. |
| **tSZ non-rel SED** | `ThermalSZSED()` | $\nu_0 = 143\,\mathrm{GHz}$ | $g(\nu) = x\coth(x/2) - 4 < 0$ for $\nu < 217\,\mathrm{GHz}$ (decrement). |
| **Calibration** | `apply_calibration` | — | `:forward` computes $c_1 c_2 C_\ell$; `:inverse` computes $\frac{1}{y_1 y_2} C_\ell$ (Plik convention). |
| **$C_\ell \leftrightarrow D_\ell$** | `dCl_dell_from_Dl` | — | Exact conversion uses $\ell(\ell+1) / 2\pi$, not $\ell^2 / 2\pi$. |

---

## Contributing

Contributions are welcome! Please ensure that new features:
1. Maintain survey-agnostic design (dispatching on mathematics, not survey names).
2. Remain type-stable and non-allocating on inner kernels.
3. Include unit tests and AD verification with ForwardDiff and Mooncake.

---

## License

This project is licensed under the MIT License.
