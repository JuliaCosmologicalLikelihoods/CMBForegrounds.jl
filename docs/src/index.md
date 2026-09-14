# CMBForegrounds.jl

*A high-performance, differentiable, survey-agnostic Julia library for CMB foreground, secondary anisotropy, and instrumental systematics modeling.*

---

## Overview & Architecture

`CMBForegrounds.jl` provides foundational building blocks for constructing Cosmic Microwave Background (CMB) likelihoods. Rather than hardcoding survey-specific recipes or survey names, the library uses **multiple dispatch over mathematical representations**:

1. **Angular Shapes** (`AbstractAngularShape`): Power laws, Poisson shot noise, tabulated templates, and tilted/running templates.
2. **Spectral Energy Distributions (SEDs)** (`AbstractSED`): Modified blackbodies (thermal dust, CIB), power-law radio synchrotron/free-free, non-relativistic thermal Sunyaev-Zel'dovich (tSZ), constant responses, and empirical unit responses (`NoSED`).
3. **Sky Components** (`SkyComponent`): Direct mathematical composition of an angular shape and an SED, evaluated at effective frequencies or integrated across passbands.
4. **Cross-Correlations** (`AbstractCorrelation`): Scale-dependent template correlations and scale-dependent geometric-mean auto-cross correlations.
5. **Passband & Chromatic Beam Integration** (`AbstractBand`): Dirac-delta bands, discretized top-hat/measured transmission curves, and frequency-dependent chromatic beam profiles.
6. **Instrumental Operations**: Map-level and spectrum-level calibration gains, additive systematic templates, map-response polarization leakage algebra, beam eigenmodes, super-sample lensing (SSL), and relativistic aberration.

```
                  ┌───────────────────────┐
                  │  AbstractAngularShape │
                  └──────────┬────────────┘
                             │
                             ▼
┌─────────────┐       ┌──────────────┐
│ AbstractSED │ ───►  │ SkyComponent │ ───► eval_component / eval_component_te
└─────────────┘       └──────────────┘
                             │
                             ▼
                  ┌───────────────────────┐
                  │   Bandpass/Chromatic  │ ───► integrate_chromatic_sed
                  └──────────┬────────────┘
                             │
                             ▼
                  ┌───────────────────────┐
                  │ Instrumental Pipeline │ ───► Calibration, Leakage, Aberration, Templates
                  └───────────────────────┘
```

> [!NOTE]
> **Survey Data Assets Remain External**
> Tabulated sky templates, measured bandpasses, beam transfer matrices, and experiment likelihood covariances are user or survey assets. Keeping them external ensures `CMBForegrounds.jl` remains lean, lightweight, and universally applicable across Planck, ACT, SPT, Simons Observatory, and CMB-S4.

---

## Mathematical Representations

### 1. Angular Shapes

Angular power spectra $D_\ell$ (or $C_\ell$) are represented by subtypes of `AbstractAngularShape`:

- `PowerLawShape(amp, alpha; ell_0=3000.0)`:
  $$D_\ell = A \left(\frac{\ell}{\ell_0}\right)^\alpha$$
- `PoissonShape(amp; ell_0=3000.0)`:
  $$D_\ell = A \left(\frac{\ell}{\ell_0}\right)^2$$
  equivalent to white shot noise with constant $C_\ell = A \frac{2\pi}{\ell_0^2}$.
- `TemplateShape(template; ell_0=3000.0, ell_min=2)`:
  Evaluates a tabulated template normalized such that $D_{\ell_0} = 1$:
  $$D_\ell = \frac{T(\ell)}{T(\ell_0)}$$
- `TiltedTemplateShape(template, tilt; ell_0=3000.0, ell_min=2)`:
  Tabulated template with power-law running tilt $\gamma$:
  $$D_\ell = \frac{T(\ell)}{T(\ell_0)} \left(\frac{\ell}{\ell_0}\right)^\gamma$$

Evaluate any shape on an array of multipoles via:
```julia
cl = angular_power(shape, ell)
```

### 2. Spectral Energy Distributions (SEDs)

Frequency responses in thermodynamic $\Delta T_{\rm CMB}$ units are represented by subtypes of `AbstractSED`:

- `ModifiedBlackbodySED(beta, T_d; nu_0=353.0)`:
  Thermal dust and Cosmic Infrared Background (CIB) emission:
  $$f(\nu) = \left(\frac{\nu}{\nu_0}\right)^\beta \frac{B_\nu(T_d)}{B_{\nu_0}(T_d)} \left[\frac{\left.\frac{\partial B_\nu}{\partial T}\right|_{T_{\rm CMB}}}{\left.\frac{\partial B_{\nu_0}}{\partial T}\right|_{T_{\rm CMB}}}\right]^{-1}$$
- `RadioSED(alpha; nu_0=150.0)`:
  Power-law in Rayleigh-Jeans temperature:
  $$f(\nu) = \left(\frac{\nu}{\nu_0}\right)^\alpha \left[\frac{\left.\frac{\partial B_\nu}{\partial T}\right|_{T_{\rm CMB}}}{\left.\frac{\partial B_{\nu_0}}{\partial T}\right|_{T_{\rm CMB}}}\right]^{-1}$$
  *(Note: Rayleigh-Jeans index $\alpha = \alpha_{\rm flux} - 2$)*.
- `ThermalSZSED(; nu_0=143.0)`:
  Non-relativistic thermal Sunyaev-Zel'dovich effect:
  $$f(\nu) = \frac{g(\nu)}{g(\nu_0)}, \quad g(x) = x \frac{e^x+1}{e^x-1} - 4, \quad x = \frac{h\nu}{k_B T_{\rm CMB}}$$
- `ConstantSED(value)`: Frequency-independent scaling factor.
- `NoSED()`: Identity response ($1.0$), enabling direct caller-specified pairwise cross-frequency amplitudes without physical SED constraints.

Evaluate any SED at frequency $\nu$ via:
```julia
w = sed_weight(sed, nu)
```

### 3. Sky Components

A physical sky component binds an angular shape with an SED:
```julia
comp = SkyComponent(shape, sed)

# Evaluate cross-frequency power spectrum:
# C_ell(nu1, nu2) = f(nu1) * f(nu2) * D_ell
cl_cross = eval_component(comp, nu1, nu2, ell)

# For polarization TE cross-spectra:
cl_te = eval_component_te(comp, nu1, nu2, ell)
```

### 4. Cross-Correlations

Scale-dependent cross-correlations between distinct components (e.g. tSZ $\times$ CIB) are modeled with `AbstractCorrelation`:

- `TemplateCorrelation(template; ell_0=3000.0, ell_min=2, xi=1.0)`:
  Scale-dependent correlation defined by an external cross-template:
  $$C_\ell^{\rm cross} = \xi \frac{T(\ell)}{T(\ell_0)}$$
- `GeometricMeanCorrelation(shape1, shape2; xi=1.0)`:
  Scale-dependent geometric mean of individual angular shapes:
  $$C_\ell^{\rm cross} = \xi \sqrt{|D_{\ell,1} D_{\ell,2}|}$$

Evaluate via:
```julia
cl = correlation_power(corr, ell)
```

---

## Bandpass & Chromatic Beams

Realistic instruments do not observe at single delta frequencies. `CMBForegrounds.jl` supports both delta and integrated bandpass transmissions:

- `DeltaBand(nu)`: Infinitesimally narrow bandpass at effective frequency $\nu$.
- `Band(nus, weights)`: Discretized transmission spectrum $\tau(\nu)$ with trapezoidal quadrature.
- `ChromaticBeam(nus, fwhm_arcmin)`: Frequency-dependent beam resolution profile $\theta_{\rm FWHM}(\nu)$.

### Integrated SED Weights

```julia
# Integrate SED across passband with optional chromatic beam
w = integrate_chromatic_sed(sed, band; beam=beam)

# Compute full cross-band SED weight matrix W[i, j] = w_i * w_j:
W = eval_chromatic_sed_bands(sed, bands; beams=beams)
```

### Optimized Matrix Cross Kernels

To evaluate multi-frequency cross-spectra without allocating per-component 3D arrays, the library provides factorized matrix kernels:
```julia
# Factorized auto/symmetric cross: C[i, j, l] = F[i, j] * cl[l]
C = factorized_cross(F, cl)

# Factorized TE cross: C[i, j, l] = FT[i] * FE[j] * cl[l]
C_te = factorized_cross_te(FT, FE, cl)
```
These kernels feature custom Mooncake adjoint registrations and ChainRules pullbacks for optimal reverse-mode automatic differentiation.

---

## Instrumental Operations

CMB systematics are modeled through modular, composable transformations:

### 1. Calibration
Supports map-level gains $c_i$ or spectrum-level calibration matrices $C_{ij}$:
```julia
# Forward convention: C_obs = c1 * c2 * C_sky
cl_cal = apply_calibration(cl, c1, c2; convention=:forward)

# Inverse convention (Planck Plik convention: C_obs = C_sky / (y1 * y2))
cl_cal = apply_calibration(cl, y1, y2; convention=:inverse)

# Multi-channel map gains or spectrum calibration matrices:
C_cal = apply_calibration(C_cube, gains; convention=:forward)
```

### 2. Additive Systematic Templates
Fixed or sampled amplitude templates:
```julia
cl_obs = add_template(cl_sky, template_arr, amp)
```

### 3. Polarization Leakage
Map-level leakage algebra for intensity-to-polarization leakage:
$$\begin{aligned}
C_\ell^{TE, \rm obs} &= C_\ell^{TE} + \frac{1}{2} \eta_T C_\ell^{TT} \\
C_\ell^{EE, \rm obs} &= C_\ell^{EE} + \eta_E C_\ell^{TE} + \frac{1}{4} \eta_E^2 C_\ell^{TT}
\end{aligned}$$
Evaluated via:
```julia
cl_te_obs = apply_te_leakage(cl_tt, cl_te, eta_T)
cl_ee_obs = apply_ee_leakage(cl_tt, cl_te, cl_ee, eta_E)
```

### 4. Beam Perturbation Eigenmodes
Linear and quadratic beam distortion modes:
```julia
cl_dist = beam_eigenmode_response(cl, modes, coeffs)
```

### 5. Super-Sample Lensing & Relativistic Aberration
```julia
# Super-sample lensing response: -kappa * d(ell^2 C_ell)/d(ell) / ell
cl_ssl = apply_ssl(cl, dcl_dell, kappa)

# Relativistic aberration dipole: -beta * d(C_ell)/d(ln ell)
cl_aberr = apply_aberration(cl, dcl_dell, beta)
```

---

## Conventions Quick-Reference

| Quantity | Representation / Function | Conventions & Details |
|:---|:---|:---|
| **Radio Index** | `RadioSED(alpha)` | Rayleigh-Jeans temperature index $\alpha = \alpha_{\rm flux} - 2$. |
| **Angular Slope** | `PowerLawShape(A, alpha)` | Evaluates $A (\ell / \ell_0)^\alpha$ directly. Legacy `dust_tt_power_law` took $\alpha + 2$. |
| **Poisson Noise** | `PoissonShape(A)` | Evaluates $A (\ell / \ell_0)^2$, equivalent to white noise $C_\ell = \text{const}$. |
| **tSZ Decrement** | `ThermalSZSED()` | $g(\nu) = x\coth(x/2) - 4 < 0$ for $\nu < 217\,\mathrm{GHz}$. |
| **Cross Symmetrization** | Cross spectra | Factor of 2 applied to cross-frequency SED combinations ($f_1 f_2' + f_1' f_2$). |
| **Calibration** | `apply_calibration` | `:forward` ($c_1 c_2$) vs `:inverse` ($1 / [y_1 y_2]$). |
| **$C_\ell \leftrightarrow D_\ell$** | `dCl_dell_from_Dl` | Exact conversion uses $\frac{\ell(\ell+1)}{2\pi}$, not $\frac{\ell^2}{2\pi}$. |

---

## Support & Verification Matrix

| Survey / Experiment Prescription | Implemented Representation | Integration Test Status | Notes |
|:---|:---|:---|:---|
| **HiLLiPoP PR4** | `TemplateShape`, `ModifiedBlackbodySED`, `RadioSED`, `TemplateCorrelation` | Verified | PR4 dust template, point bands, tSZxCIB cross template |
| **SPT2018 / D1** | `PoissonShape`, `PowerLawShape`, `GeometricMeanCorrelation`, `beam_eigenmode_response` | Verified | Clustered CIB, geometric tSZ-CIB, beam modes, map gains |
| **ACT DR4** | `Band`, `ChromaticBeam`, `integrate_chromatic_sed`, `TiltedTemplateShape` | Verified | Bandpass integration, chromatic beams, tSZ tilt |
| **Planck Plik** | `PowerLawShape`, `apply_calibration(:inverse)`, `apply_te_leakage`, `apply_aberration` | Verified | Map gains, leakage algebra, relativistic aberration |
| **CamSpec** | `ConstantSED`, `NoSED`, spectrum calibration matrix | Verified | Pairwise empirical power laws, 2D calibration matrix |
| **Assemblers** | `assemble_TT`, `assemble_EE`, `assemble_TE` | Parity & AD Verified | Full parity with Phase 0 baseline; Mooncake & ForwardDiff tested |
| **4-point Covariance** | *(Deferred)* | Intentionally limited | Non-Gaussian trispectrum covariances remain likelihood-owned |

---

## Documentation Index

```@index
```