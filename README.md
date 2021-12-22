# ![](mtalg/__res/_MTA.png) *mtalg* — Multithreaded Algebra 


[![CircleCI](https://circleci.com/gh/LucaMingarelli/CryptPandas.svg?style=svg&circle-token=23ad3dc02a697420107b82330e00b944f9ea53ed)]()
[![version](https://img.shields.io/badge/version-0.0.6-success.svg)](#)

[//]: # ([![PyPI Latest Release]&#40;https://img.shields.io/pypi/v/CryptPandas.svg&#41;]&#40;https://pypi.org/project/CryptPandas/&#41;)
[![License](https://img.shields.io/pypi/l/CryptPandas.svg)](https://github.com/WWakker/mtalg/blob/master/LICENSE.txt)

[//]: # ([![Downloads]&#40;https://static.pepy.tech/personalized-badge/cryptpandas?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads&#41;]&#40;https://pepy.tech/project/cryptpandas&#41;)

[//]: # ([![Build and test GitHub]&#40;https://github.com/lucamingarelli/CryptPandas/actions/workflows/build-and-test.yml/badge.svg&#41;]&#40;https://github.com/LucaMingarelli/CryptPandas/actions&#41;)



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

Major benefits become apparent past $`10^7`$ operations for both the element-wise algebra and the random number generator modules.

# Installation

`pip install mtalg`

# How to use
Import random number generator and algebra functions
```python
import mtalg
from mtalg.random import MultithreadedRNG
```
Create an instance of the multithreaded random number generator with seed for reproducability and number of threads to be used
```python
mrng = MultithreadedRNG(seed=1, num_threads=4)
```
Create two arrays (results are stored in `mrng.values`)
```python
mrng.standard_normal(size=(10_000, 5_000))
A = mrng.values
mrng.uniform(size=(10_000, 5_000), low=0, high=10)
B = mrng.values
```
Add B to A (A is modified inplace)
```python
mtalg.add(A, B, num_threads=4)
```
Subtract A from B (B is modified inplace)
```python
mtalg.sub(A, B, direction='right', num_threads=4)
```
Multiply, divide and raise to power (A is modified inplace)
```python
mtalg.mul(A, B, num_threads=4)
mtalg.div(A, B, num_threads=4)
mtalg.pow(A, B, num_threads=4)
```

# Benchmarks

### Elementwise algebra
![](mtalg/__res/benchmark/benchmark_add_BARS.svg)

![](mtalg/__res/benchmark/benchmark_add.svg)


### Random number generation

![](mtalg/__res/benchmark/benchmark_rng_BAR.svg)

![](mtalg/__res/benchmark/benchmark_rng.svg)



# Aknowledgments

The module for multithreaded generation of random numbers is inspired from [here](https://numpy.org/doc/stable/reference/random/multithreading.html).  

# Authors
[Wouter Wakker](https://github.com/WWakker) 
and [Luca Mingarelli](https://github.com/LucaMingarelli), 
2021

[![Python](https://img.shields.io/static/v1?label=made%20with&message=Python&color=blue&style=for-the-badge&logo=Python&logoColor=white)](#)
