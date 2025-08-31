# CMBForegrounds.jl

| **Documentation** | **Build Status** | **Code Coverage** | **Code Style** |
|:--------:|:----------------:|:----------------:|:----------------:|
| [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliacosmologicallikelihoods.github.io/CMBForegrounds.jl/dev) [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliacosmologicallikelihoods.github.io/CMBForegrounds.jl/stable) | [![Build status (Github Actions)](https://github.com/JuliaCosmologicalLikelihoods/CMBForegrounds.jl/workflows/CI/badge.svg)](https://github.com/JuliaCosmologicalLikelihoods/CMBForegrounds.jl/actions) | [![codecov](https://codecov.io/gh/JuliaCosmologicalLikelihoods/CMBForegrounds.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaCosmologicalLikelihoods/CMBForegrounds.jl) | [![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle) [![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac) |

`CMBForegrounds.jl` is a high-performance, differentiable Julia package for computing foreground components of the Cosmic Microwave Background (CMB) power spectrum.

This package is designed as a foundational toolkit for building CMB likelihoods. It is **agnostic** by design: no specific experimental templates are included. Instead, it provides the core spectral and spatial models, allowing users to construct likelihoods for various experiments by providing templates where necessary.

The core design principles are:
* 🚀 **Performance:** All functions are type-stable and optimized for speed.
* 🧠 **Differentiability:** The entire model pipeline is compatible with automatic differentiation (AD), enabling gradient-based parameter inference.
* 🛠️ **Modularity:** The functions are self-contained and can be easily composed to build complex foreground models.

---

## ✨ Features

-   A comprehensive suite of foreground models:
    -   Thermal and Kinematic Sunyaev-Zel'dovich (tSZ, kSZ) effects
    -   Cosmic Infrared Background (CIB) clustered and shot-noise components
    -   Thermal dust emission
    -   tSZ-CIB cross-correlation
-   Models for physical effects on power spectra:
    -   Super-sample lensing (SSL)
    -   Relativistic aberration
-   Core physics utilities for blackbody radiation, spectral energy distribution (SED) weighting, and beam effects.
-   Clean, tested, and well-documented code.

---

## 🤝 Contributing

Contributions are welcome! If you find a bug, have a feature request, or want to contribute code, please feel free to contac us and/or open an issue or submit a pull request on the GitHub repository.

---

## 📜 License

This package is licensed under the MIT License. See the `LICENSE` file for more details.
