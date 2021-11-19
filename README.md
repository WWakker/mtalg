# ![](mtalg/__res/_MTA.png) *mtalg* — Multithreaded Algebra 

[![version](https://img.shields.io/badge/version-0.0.2-success.svg)](#)

# About

***mtalg*** is a package for multithreaded algebra and random number generation.

While numpy does support out of the box multithreaded linear algebra 
([numpy.linalg](https://numpy.org/doc/stable/reference/routines.linalg.html)) 
for operations such as matrix multiplication, decomposition, spectral analysis, 
and related functions, which are building on libraries such as BLAS and LAPACK, 
the same does not hold true for simpler algebraic element-wise operations. 
Similarly can be said for the generation of random variates.

***mtalg*** is the fastest library known to us for large scale element-wise algebraic operations 
and random number generation. For more info on benchmarks you can see the dedicated section below. 

Major benefits become apparent past $`10^7`$ operations for the element-wise algebra module, 
and for more than XX operations for the random number generator module.

# Installation

You can simply install from the ECB artifactory via pip as:

`pip install ecb-mtalg`

# How to use

xxx

# Benchmarks

### Elementwise algebra
![](mtalg/__res/benchmark/benchmark_add_BARS.svg)

![](mtalg/__res/benchmark/benchmark_add.svg)


### Random number generation

![](mtalg/__res/benchmark/benchmark_rng_BARS.svg)

![](mtalg/__res/benchmark/benchmark_rng.svg)



# Aknowledgments

The module for multithreaded generation of random numbers is inspired from [here](https://numpy.org/doc/stable/reference/random/multithreading.html).  

# Authors
[Wouter Wakker](https://gitlab.sofa.dev/Wouter.Wakker) 
and [Luca Mingarelli](https://gitlab.sofa.dev/Luca.Mingarelli), 
2021

[![Python](https://img.shields.io/static/v1?label=made%20with&message=Python&color=blue&style=for-the-badge&logo=Python&logoColor=white)](#)
