# CMBForegrounds.jl

*A general-purpose Julia package for building CMB likelihoods with foreground and systematic effect modeling*

## Overview

CMBForegrounds.jl is designed as a foundational tool for constructing Cosmic Microwave Background (CMB) likelihoods. The package provides well-tested, optimized, and differentiable functions for modeling the most relevant and commonly employed functional forms for:

- **Astrophysical foregrounds** (galactic dust, synchrotron, free-free emission)
- **Secondary anisotropies** (thermal and kinematic Sunyaev-Zel'dovich effects)
- **Instrumental systematics** and calibration effects
- **Spectral energy distributions** and frequency-dependent responses

## Key Features

- **Well-tested**: Comprehensive test suite with 158+ unit tests ensuring numerical accuracy
- **Optimized**: Efficient implementations of spectral functions and frequency transformations
- **Differentiable**: Compatible with automatic differentiation frameworks for gradient-based inference
- **General-purpose**: Flexible building blocks for custom likelihood implementations
- **Standards-compliant**: Follows Julia community best practices and conventions

## Core Functionality

The package provides fundamental spectral functions including:

- Planck function ratios and temperature derivatives
- Thermal Sunyaev-Zel'dovich (tSZ) spectral functions
- Dimensionless frequency variables for blackbody calculations
- Physical constants and unit conversions

## Installation

```julia
using Pkg
Pkg.add("CMBForegrounds")
```

## Physical Constants

The package includes several important physical constants used in CMB analysis:

- `T_CMB`: CMB temperature (2.72548 K)
- `h`: Planck constant 
- `kB`: Boltzmann constant
- `Ghz_Kelvin`: Conversion factor h/kB × 10⁹

## Documentation

```@index
```