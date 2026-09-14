# CMBForegrounds.jl

`CMBForegrounds.jl` is a small library of differentiable mathematical operations used to construct multifrequency CMB foreground models. It does not define complete Planck, ACT, or SPT likelihoods.

## Scope

The shared package owns reusable mathematics: angular shapes, SED and passband evaluation, cross-spectrum assembly, selected instrumental operations, and efficient fused foreground kernels.

Likelihood packages own numerical assets, parameter names and priors, map/mask selection, data-vector ordering, binning, baseline covariance, and the decision whether an effect is fitted, fixed, already corrected, or absent.

## Angular models

```julia
using CMBForegrounds

ells = collect(30:3000)
power_law = PowerLawShape(500.0)
D_power = angular_power(power_law, ells; amp=8.0, alpha=-0.6)

poisson = PoissonShape(3000.0)
D_poisson = angular_power(poisson, ells; amp=3.0)

dense_template = ones(3001) # samples ell=0,...,3000
template = TemplateShape(dense_template; ell_0=3000, ell_min=0)
D_template = angular_power(template, ells; amp=4.0)

tilted = TiltedTemplateShape(template, 3000.0)
D_tilted = angular_power(tilted, ells; amp=4.0, alpha=-0.2)
```

`TemplateShape` uses a dense integer multipole grid. Setting `ell_0=nothing` declares that the supplied template is already normalized.

## SEDs and components

```julia
dust_sed = ModifiedBlackbodySED(150.0, 19.6)
dust = SkyComponent(dust_sed, PowerLawShape(500.0))
D_dust = eval_component(dust, ells, 90.0, 150.0, 8.0, 1.5; alpha=-0.6)

radio_sed = RadioSED(150.0; convention=:flux)
radio = SkyComponent(radio_sed, PoissonShape(3000.0))
D_radio = eval_component(radio, ells, 90.0, 150.0, 3.0, -0.7)
```

The `SkyComponent` constructor takes `(sed, angular)`. Amplitudes, slopes and SED indices are evaluation arguments rather than fields, leaving parameter sharing to the caller.

`ConstantSED()` represents a frequency-independent thermodynamic signal such as kSZ. `NoSED()` is the same numerical response with distinct semantics for empirical residuals that should not be assigned a physical SED.

## Passband integration

Use `DeltaBand(nu)` for an effective frequency. Use `make_band(nu, transmission)` for a tabulated response. For differentiable shifts, retain the unnormalized response in `RawBand` and call `shift_and_normalize`.

Chromatic responses are supplied as explicit matrices:

```julia
nu = collect(range(130.0, 170.0; length=41))
transmission = @. exp(-0.5 * ((nu - 150.0) / 10.0)^2)
band = make_band(nu, transmission)

beam_values = [1 + 0.01 * (l / 3000) * (n - 150) / 20
               for l in ells, n in nu]
beam = ChromaticBeam(ells, beam_values)
weights = sed_weight(dust_sed, band, beam, 1.5)
```

The beam and requested component must use the same multipole grid.

## Component correlations

`TemplateCorrelation` implements a signed, symmetrized cross-template prescription. `GeometricMeanCorrelation` implements the distinct geometric-mean construction.

```julia
cross = TemplateCorrelation(template)
D_cross = correlation_power(
    cross, ells, xi, A_tsz, A_cib,
    f_tsz_i, f_tsz_j, f_cib_i, f_cib_j,
)
```

The frequency factor contains both leg orderings. At equal frequencies this naturally produces a factor of two. The amplitude domain and prior on `xi` remain likelihood policy.

## Instrumental operations

- `apply_calibration` supports map gains and supplied pair-gain matrices, with explicit forward/inverse conventions.
- `add_template` adds supplied systematic templates.
- `te_leakage`, `et_leakage`, and `ee_leakage` expose T-to-E map-response algebra while retaining ordered TE and ET spectra.
- `beam_eigenmode_response` and `beam_eigenmode_cross` apply supplied beam modes.
- `apply_ssl` and `apply_aberration` add the corresponding response to an input spectrum.

These functions do not decide whether an uncertainty belongs in the mean or covariance. They also do not recreate effects already corrected in released data.

## Spectrum conventions

All new angular models evaluate ``D_ell = ell(ell+1)C_ell/(2pi)``.

- `PowerLawShape` takes a `D_ell` slope directly.
- `PoissonShape` is exactly constant in `C_ell`, hence proportional to `ell(ell+1)` in `D_ell`.
- Legacy `shot_noise_power` preserves its intentional `ell^2` convention.
- Radio flux and RJ indices satisfy `beta_rj = beta_flux - 2`.

## Validation boundary

Package tests establish algebraic behavior, AD agreement, and compatibility with legacy kernels. Synthetic survey-style tests exercise composition only. Exact agreement with released survey likelihoods must be tested downstream using each survey's templates, passbands, corrections, windows, and parameter conventions.

```@index
```
