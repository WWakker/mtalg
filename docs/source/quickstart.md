# Quick start guide

***mtalg*** is a package for multithreaded algebra and random number generation.

While numpy does support out of the box multithreaded linear algebra 
([numpy.linalg](https://numpy.org/doc/stable/reference/routines.linalg.html)) 
for operations such as matrix multiplication, decomposition, spectral analysis, 
and related functions, which are building on libraries such as BLAS and LAPACK, 
the same does not hold true for simpler algebraic element-wise operations. 
Similarly can be said for the generation of random variates.

***mtalg*** is the fastest library known to us for large scale element-wise algebraic operations 
and non-GPU-based random number generation. For more info on benchmarks you can see the dedicated section below. 

Major benefits become apparent past 10⁷ operations for both the element-wise algebra and the random number generator modules.

## Installation
The library can be install via pip as:

```shell
pip install mtalg
```

## How to use
Import mtalg and generate pseudorandom numbers as

```python
import mtalg

a = mtalg.random.standard_normal(size=(10_000, 5_000))
b = mtalg.random.uniform(size=(10_000, 5_000), low=0, high=10)
# etc.
```

Alternatively, one can also
```python
from mtalg.random import MultithreadedRNG
```
and create an instance of the multithreaded random number generator with seed for reproducibility and set the number of threads to be used
```python
mrng = MultithreadedRNG(seed=1, num_threads=4)
```
One can then create random arrays as
```python
a = mrng.standard_normal(size=(10_000, 5_000))
b = mrng.uniform(size=(10_000, 5_000), low=0, high=10)
# etc.
```
Set number of threads to be used by default for algebra functions and subsquent random
number generators (if `num_threads` parameter is not specified)
```python
mtalg.set_num_threads(4)
```
Add `b` to `a` (`a` is modified in-place)
```python
mtalg.add(a, b)
```
Subtract `a` from `b` (`b` is modified in-place)
```python
mtalg.sub(a, b, direction='right')
```
Multiply, divide and raise to power (`a` is modified in-place)
```python
mtalg.mul(a, b)
mtalg.div(a, b)
mtalg.pow(a, b)
```

## Benchmarks

### Elementwise algebra
![](https://github.com/WWakker/mtalg/raw/master/mtalg/__res/benchmark/benchmark_add_BARS.svg)

![](https://github.com/WWakker/mtalg/raw/master/mtalg/__res/benchmark/benchmark_add.svg)

### Random number generation

![](https://github.com/WWakker/mtalg/raw/master/mtalg/__res/benchmark/benchmark_rng_BAR.svg)

![](https://github.com/WWakker/mtalg/raw/master/mtalg/__res/benchmark/benchmark_rng.svg)

\* Benchmarks are carrried out using an Intel(R) Xeon(R) Gold 6142M CPU @ 2.60GHz and 24 threads
