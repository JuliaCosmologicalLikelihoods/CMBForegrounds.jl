# CMBForegrounds.jl

*A Julia package for Cosmic Microwave Background foreground calculations*

## Overview

CMBForegrounds.jl provides functions for computing various spectral properties of CMB foregrounds, including:

- Planck function ratios
- Temperature derivatives of Planck functions  
- Thermal Sunyaev-Zel'dovich (tSZ) spectral functions
- Dimensionless frequency variables for blackbody calculations

## Installation

```julia
using Pkg
Pkg.add("CMBForegrounds")
```

## Quick Start

```julia
using CMBForegrounds

# Calculate dimensionless frequency variables
ν, ν0, T = 100.0, 143.0, 2.725  # frequencies in GHz, temperature in K
r, x, x0 = dimensionless_freq_vars(ν, ν0, T)

# Planck function ratio
bnu_ratio = Bnu_ratio(ν, ν0, T)

# Temperature derivative ratio  
dbdt_ratio = dBdT_ratio(ν, ν0, T)

# tSZ spectral function ratio
tsz_ratio = tsz_g_ratio(ν, ν0, T)
```

## Physical Constants

The package includes several important physical constants:

- `T_CMB`: CMB temperature (2.72548 K)
- `h`: Planck constant 
- `kB`: Boltzmann constant
- `Ghz_Kelvin`: Conversion factor h/kB × 10⁹

## Documentation

```@index
```