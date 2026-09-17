# API Reference

## Angular Power Spectrum Representations

```@docs
AbstractAngularModel
PowerLawShape
PoissonShape
TemplateShape
TiltedTemplateShape
angular_power
```

## Spectral Energy Distributions (SEDs)

```@docs
AbstractSED
ModifiedBlackbodySED
RadioSED
ThermalSZSED
ConstantSED
NoSED
SkyComponent
sed_weight
eval_component
eval_component_te
```

## Cross-Correlation Representations

```@docs
AbstractCorrelationModel
TemplateCorrelation
GeometricMeanCorrelation
correlation_power
```

## Passband & Chromatic Beam Integration

```@docs
AbstractBand
DeltaBand
Band
ChromaticBeam
integrate_sed
integrate_chromatic_sed
eval_chromatic_sed_bands
factorized_cross
factorized_cross_te
```

## Instrumental Systematic Operations

```@docs
calibration_factor
apply_calibration
additive_template
add_template
te_leakage
et_leakage
ee_leakage
apply_te_leakage
apply_ee_leakage
beam_eigenmode_response
beam_eigenmode_cross
apply_ssl
apply_aberration
```

## Multi-Channel Spectrum Assemblers

```@docs
assemble_TT
assemble_EE
assemble_TE
```

## Low-Level Physics & Spectral Functions

```@docs
dimensionless_freq_vars
Bnu_ratio
dBdT_ratio
tsz_g_ratio
tsz_cross_power
tsz_cib_cross_power
cib_mbb_sed_weight
dust_tt_power_law
cib_clustered_power
ksz_template_scaled
ssl_response
aberration_response
gaussian_beam_window
fwhm_arcmin_to_sigma_rad
shot_noise_power
dCl_dell_from_Dl
cross_calibration_mean
```

## Additional Utilities and Legacy Kernels

```@docs
RawBand
PreparedChromaticBandpass
make_band
point_band
shift_and_normalize
prepare_chromatic_bandpass
eval_sed_bands
integrate_tsz
trapz
x_cmb
cmb2bb
rj2cmb
tsz_f
tsz_sed
mbb_sed
radio_sed
constant_sed
eval_template
eval_template_tilt
eval_powerlaw
cib_clustered_template_power
tsz_cib_template_power
dust_model_template_power
radio_ps_power
dusty_ps_power
sub_pixel_power
correlated_cross
build_szxcib_cl
CMBForegrounds._radio_sed_ratio
```

## Physical Constants

```@docs
CMBForegrounds.T_CMB
CMBForegrounds.h
CMBForegrounds.kB
CMBForegrounds.Ghz_Kelvin
```

## Index

```@index
```
