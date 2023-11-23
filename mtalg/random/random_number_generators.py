"""
The docstrings in this file are based on docstrings of NumPy, which has the following licence:
----------------------------------------------------------------------------------------------

Copyright (c) 2005-2021, NumPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    * Neither the name of the NumPy Developers nor the names of any
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from numpy.random import SeedSequence, PCG64, Generator, BitGenerator
import concurrent.futures
import numpy as np
import mtalg.core.threads
from typing import Optional, Union
from numbers import Number

argmax = lambda iterable: max(enumerate(iterable), key=lambda x: x[1])[0]


class MultithreadedRNG:
    """Multithreaded random number generator.

    Args:
        seed (int): Random seed.
        num_threads (int): Number of threads to be used, overrides threads as set by :func:`~mtalg.set_num_threads`.
        bit_generator (np.random.BitGenerator): Bit generator, defaults to PCG64.

    Examples:

        Instantiate a multithreaded random number generator which uses 4 threads, setting a seed to derive the
        initial BitGenerator state.

        >>> from mtalg.random import MultithreadedRNG
        >>> mrng = MultithreadedRNG(seed=1, num_threads=4)

        Create a 10000 x 5000 matrix with numbers, drawing from the standard normal distribution.

        >>> a = mrng.standard_normal(size=(10_000, 5_000))

        Create a 10000 x 5000 matrix with numbers, drawing from the uniform distribution.

        >>> b = mrng.uniform(size=(10_000, 5_000), low=0, high=10)

    Note:
        For more information on distributions, see
        `Random generator distributions <https://numpy.org/doc/stable/reference/random/generator.html#distributions>`_.
    """

    def __init__(self, seed: Optional[int] = None, num_threads: Optional[int] = None, bit_generator: BitGenerator = PCG64):
        self._num_threads = num_threads or mtalg.core.threads._global_num_threads
        assert self._num_threads > 0 and isinstance(self._num_threads, int), \
            f'Number of threads must be an integer > 0, found: {self._num_threads}'
        seq = SeedSequence(seed)
        self._random_generators = [Generator(bit_generator=bit_generator(s)) for s in seq.spawn(self._num_threads)]
        self._shape = None
        self._shp_max = None
        self._values = None
        self._steps = []

    def beta(self, a: Number, b: Number, size: Union[int, tuple, None] = None) -> Union[np.ndarray, Number]:
        """Draw from a beta distribution.

        Args:
            a: Alpha, positive (>0).
            b: Beta, positive (>0).
            size: Output shape.

        Returns
            numpy.ndarray or scalar.
        """
        self._check_shape(size)
        kw_args = {'a': a,
                   'b': b}

        if size is None:
            return self._random_generators[0].beta(**kw_args)

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self._shp_max +
                (slice(first, last),)] = random_state.beta(size=self._get_slice_size(first, last), **kwargs)

        return self._fill(__fill, **kw_args)

    def binomial(self, n: Number, p: Number, size: Union[int, tuple, None] = None) -> Union[np.ndarray, Number]:
        """Draw from a binomial distribution.

        Args:
            n: Parameter of the distribution, >= 0. Floats are also accepted, but they will be truncated to integers.
            p: Parameter of the distribution, >= 0 and <=1.
            size: Output shape.

        Returns:
            numpy.ndarray or scalar.
        """
        self._check_shape(size)
        kw_args = {'n': n,
                   'p': p}

        if size is None:
            return self._random_generators[0].binomial(**kw_args)

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self._shp_max +
                (slice(first, last),)] = random_state.binomial(size=self._get_slice_size(first, last), **kwargs)

        return self._fill(__fill, **kw_args)

    def chisquare(self, df: Number, size: Union[int, tuple, None] = None) -> Union[np.ndarray, Number]:
        """Draw from a chisquare distribution.

        Args:
            df: Number of degrees of freedom, must be > 0.
            size: Output shape.

        Returns:
            numpy.ndarray or scalar.
        """
        self._check_shape(size)
        kw_args = {'df': df}

        if size is None:
            return self._random_generators[0].chisquare(**kw_args)

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self._shp_max +
                (slice(first, last),)] = random_state.chisquare(size=self._get_slice_size(first, last), **kwargs)

        return self._fill(__fill, **kw_args)

    def exponential(self, scale: Number = 1.0, size: Union[int, tuple, None] = None) -> Union[np.ndarray, Number]:
        """Draw from an exponential distribution.

        Args:
            scale: The scale parameter, β = 1/λ Must be non-negative.
            size: Output shape.

        Returns:
            numpy.ndarray or scalar.
        """
        self._check_shape(size)
        kw_args = {'scale': scale}

        if size is None:
            return self._random_generators[0].exponential(**kw_args)

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self._shp_max +
                (slice(first, last),)] = random_state.exponential(size=self._get_slice_size(first, last), **kwargs)

        return self._fill(__fill, **kw_args)

    def f(self, dfnum: Number, dfden: Number, size: Union[int, tuple, None] = None) -> Union[np.ndarray, Number]:
        """Draw from an F distribution.

        Args:
            dfnum: Degrees of freedom in numerator, must be > 0.
            dfden: Degrees of freedom in denominator, must be > 0.
            size: Output shape.

        Returns:
            ndarray or scalar.
        """
        self._check_shape(size)
        kw_args = {'dfnum': dfnum,
                   'dfden': dfden}

        if size is None:
            return self._random_generators[0].f(**kw_args)

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self._shp_max +
                (slice(first, last),)] = random_state.f(size=self._get_slice_size(first, last), **kwargs)

        return self._fill(__fill, **kw_args)

    def gamma(self, shape: Number, scale: Number = 1.0, size: Union[int, tuple, None] = None) -> Union[np.ndarray, Number]:
        """Draw from a gamma distribution.

        Args:
            shape: The shape of the gamma distribution. Must be non-negative.
            scale: The scale of the gamma distribution. Must be non-negative. Default is equal to 1.
            size: Output shape.

        Returns:
            numpy.ndarray or scalar.
        """
        self._check_shape(size)
        kw_args = {'shape': shape,
                   'scale': scale}

        if size is None:
            return self._random_generators[0].gamma(**kw_args)

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self._shp_max +
                (slice(first, last),)] = random_state.gamma(size=self._get_slice_size(first, last), **kwargs)

        return self._fill(__fill, **kw_args)

    def geometric(self, p: Number, size: Union[int, tuple, None] = None) -> Union[np.ndarray, Number]:
        """Draw from a geometric distribution.

        Args:
            p: The probability of success of an individual trial.
            size: Output shape.

        Returns:
            numpy.ndarray or scalar.
        """
        self._check_shape(size)
        kw_args = {'p': p}

        if size is None:
            return self._random_generators[0].geometric(**kw_args)

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self._shp_max +
                (slice(first, last),)] = random_state.geometric(size=self._get_slice_size(first, last), **kwargs)

        return self._fill(__fill, **kw_args)

    def gumbel(self, loc: Number = 0.0, scale: Number = 1.0, size: Union[int, tuple, None] = None) -> Union[np.ndarray, Number]:
        """Draw from a Gumbel distribution.

        Args:
            loc: The location of the mode of the distribution. Default is 0.
            scale: The scale parameter of the distribution. Default is 1. Must be non-negative.
            size: Output shape.

        Returns:
            numpy.ndarray or scalar.
        """
        self._check_shape(size)
        kw_args = {'loc': loc,
                   'scale': scale}

        if size is None:
            return self._random_generators[0].gumbel(**kw_args)

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self._shp_max +
                (slice(first, last),)] = random_state.gumbel(size=self._get_slice_size(first, last), **kwargs)

        return self._fill(__fill, **kw_args)

    def hypergeometric(self, ngood: int, nbad: int, nsample: int, size: Union[int, tuple, None] = None) -> Union[np.ndarray, Number]:
        """Draw from a hypergeometric distribution.

        Args:
            ngood: Number of ways to make a good selection. Must be nonnegative and less than 10**9.
            nbad: Number of ways to make a bad selection. Must be nonnegative and less than 10**9.
            nsample: Number of items sampled. Must be nonnegative and less than ngood + nbad.
            size: Output shape.

        Returns:
            numpy.ndarray or scalar.
        """
        self._check_shape(size)
        kw_args = {'ngood': ngood,
                   'nbad': nbad,
                   'nsample': nsample}

        if size is None:
            return self._random_generators[0].hypergeometric(**kw_args)

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self._shp_max +
                (slice(first, last),)] = random_state.hypergeometric(size=self._get_slice_size(first, last), **kwargs)

        return self._fill(__fill, **kw_args)

    def integers(self, low: int, high: int = None, size: Union[int, tuple, None] = None, dtype=np.int64, endpoint: bool = False) -> Union[np.ndarray, Number]:
        """Draw random integers from low (inclusive) to high (exclusive), or if endpoint=True, low (inclusive) to high
        (inclusive).

        Args:
            low: Lowest (signed) integers to be drawn from the distribution (unless high=None, in which case this
                parameter is 0 and this value is used for high).
            high: If provided, one above the largest (signed) integer to be drawn from the distribution (see above for
                behavior if high=None).
            size: Output shape.
            dtype: Dtype of output array.
            endpoint: If true, sample from the interval [low, high] instead of the default [low, high);
                defaults to False.

        Returns:
            numpy.ndarray or scalar.
        """
        self._check_shape(size)
        kw_args = {'low': low,
                   'high': high,
                   'dtype': dtype,
                   'endpoint': endpoint}

        if size is None:
            return self._random_generators[0].integers(**kw_args)

        if self._values.dtype != dtype:
            self._values = self._values.astype(dtype)

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self._shp_max +
                (slice(first, last),)] = random_state.integers(size=self._get_slice_size(first, last), **kwargs)

        return self._fill(__fill, **kw_args)

    def laplace(self, loc: Number = 0.0, scale: Number = 1.0, size: Union[int, tuple, None] = None) -> Union[np.ndarray, Number]:
        """Draw from a Laplace distribution.

        Args:
            loc: The position, μ, of the distribution peak. Default is 0.
            scale: λ, the exponential decay. Default is 1. Must be non- negative.
            size: Output shape.

        Returns:
            numpy.ndarray or scalar.
        """
        self._check_shape(size)
        kw_args = {'loc': loc,
                   'scale': scale}

        if size is None:
            return self._random_generators[0].laplace(**kw_args)

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self._shp_max +
                (slice(first, last),)] = random_state.laplace(size=self._get_slice_size(first, last),
                                                              **kwargs)

        return self._fill(__fill, **kw_args)

    def logistic(self, loc: Number = 0.0, scale: Number = 1.0, size: Union[int, tuple, None] = None) -> Union[np.ndarray, Number]:
        """Draw from a logistic distribution.

        Args:
            loc: Parameter of the distribution. Default is 0.
            scale: Parameter of the distribution. Must be non-negative. Default is 1.
            size: Output shape.

        Returns:
            numpy.ndarray or scalar.
        """
        self._check_shape(size)
        kw_args = {'loc': loc,
                   'scale': scale}

        if size is None:
            return self._random_generators[0].logistic(**kw_args)

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self._shp_max +
                (slice(first, last),)] = random_state.logistic(size=self._get_slice_size(first, last),
                                                               **kwargs)

        return self._fill(__fill, **kw_args)

    def lognormal(self, mean: Number = 0.0, sigma: Number = 1.0, size: Union[int, tuple, None] = None) -> Union[np.ndarray, Number]:
        """Draw from a lognormal distribution.

        Args:
            mean: Mean value of the underlying normal distribution. Default is 0.
            sigma: Standard deviation of the underlying normal distribution. Must be non-negative. Default is 1.
            size: Output shape.

        Returns:
            numpy.ndarray or scalar.
        """
        self._check_shape(size)
        kw_args = {'mean': mean,
                   'sigma': sigma}

        if size is None:
            return self._random_generators[0].lognormal(**kw_args)

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self._shp_max +
                (slice(first, last),)] = random_state.lognormal(size=self._get_slice_size(first, last),
                                                                **kwargs)

        return self._fill(__fill, **kw_args)

    def logseries(self, p: Number, size: Union[int, tuple, None] = None) -> Union[np.ndarray, Number]:
        """Draw from a logarithmic series distribution.

        Args:
            p: Shape parameter for the distribution. Must be in the range (0, 1).
            size: Output shape.
        """
        self._check_shape(size)
        kw_args = {'p': p}

        if size is None:
            return self._random_generators[0].logseries(**kw_args)

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self._shp_max +
                (slice(first, last),)] = random_state.logseries(size=self._get_slice_size(first, last), **kwargs)

        return self._fill(__fill, **kw_args)

    def negative_binomial(self, n: Number, p: Number, size: Union[int, tuple, None] = None) -> Union[np.ndarray, Number]:
        """Draw from a negative binomial distribution.

        Args:
            n: Parameter of the distribution, > 0.
            p: Parameter of the distribution. Must satisfy 0 < p <= 1.
            size: Output shape.

        Returns:
            numpy.ndarray or scalar.
        """
        self._check_shape(size)
        kw_args = {'n': n,
                   'p': p}

        if size is None:
            return self._random_generators[0].negative_binomial(**kw_args)

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self._shp_max +
                (slice(first, last),)] = random_state.negative_binomial(size=self._get_slice_size(first, last),
                                                                        **kwargs)

        return self._fill(__fill, **kw_args)

    def noncentral_chisquare(self, df: Number, nonc: Number, size: Union[int, tuple, None] = None) -> Union[np.ndarray, Number]:
        """Draw from a noncentral chisquare distribution.

        Args:
            df: Number of degrees of freedom, must be > 0.
            nonc: Non-centrality, must be non-negative.
            size: Output shape.

        Returns:
            numpy.ndarray or scalar.
        """
        self._check_shape(size)
        kw_args = {'df': df,
                   'nonc': nonc}

        if size is None:
            return self._random_generators[0].noncentral_chisquare(**kw_args)

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self._shp_max +
                (slice(first, last),)] = random_state.noncentral_chisquare(size=self._get_slice_size(first, last),
                                                                           **kwargs)

        return self._fill(__fill, **kw_args)

    def noncentral_f(self, dfnum: Number, dfden: Number, nonc: Number, size: Union[int, tuple, None] = None) -> Union[np.ndarray, Number]:
        """Draw from a noncentral F distribution.

        Args:
            dfnum: Degrees of freedom in numerator, must be > 0.
            dfden: Degrees of freedom in denominator, must be > 0.
            nonc: Non-centrality parameter, the sum of the squares of the numerator means, must be >= 0.
            size: Output shape.

        Returns:
            numpy.ndarray or scalar.
        """
        self._check_shape(size)
        kw_args = {'dfnum': dfnum,
                   'dfden': dfden,
                   'nonc': nonc}

        if size is None:
            return self._random_generators[0].noncentral_f(**kw_args)

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self._shp_max +
                (slice(first, last),)] = random_state.noncentral_f(size=self._get_slice_size(first, last), **kwargs)

        return self._fill(__fill, **kw_args)

    def normal(self, loc: Number = 0.0, scale: Number = 1.0, size: Union[int, tuple, None] = None) -> Union[np.ndarray, Number]:
        """Draw from the normal distribution.

        Args:
            loc: Mean of the distribution.
            scale: Standard deviation of the distribution.
            size: Output shape.

        Returns:
            numpy.ndarray or scalar.
        """
        self._check_shape(size)
        kw_args = {'loc': loc,
                   'scale': scale}

        if size is None:
            return self._random_generators[0].normal(**kw_args)

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self._shp_max +
                (slice(first, last),)] = random_state.normal(size=self._get_slice_size(first, last), **kwargs)

        return self._fill(__fill, **kw_args)

    def pareto(self, a: Number, size: Union[int, tuple, None] = None) -> Union[np.ndarray, Number]:
        """Draw from a Pareto II or Lomax distribution.

        Args:
            a: Shape of the distribution. Must be positive.
            size: Output shape.

        Returns:
            numpy.ndarray or scalar.
        """
        self._check_shape(size)
        kw_args = {'a': a}

        if size is None:
            return self._random_generators[0].pareto(**kw_args)

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self._shp_max +
                (slice(first, last),)] = random_state.pareto(size=self._get_slice_size(first, last), **kwargs)

        return self._fill(__fill, **kw_args)

    def poisson(self, lam: Number = 1.0, size: Union[int, tuple, None] = None) -> Union[np.ndarray, Number]:
        """Draw from a poisson distribution.

        Args:
            lam: Expected number of events occurring in a fixed-time interval, must be >= 0.
            size: Output shape.

        Returns:
            numpy.ndarray or scalar.
        """
        self._check_shape(size)
        kw_args = {'lam': lam}

        if size is None:
            return self._random_generators[0].poisson(**kw_args)

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self._shp_max +
                (slice(first, last),)] = random_state.poisson(size=self._get_slice_size(first, last), **kwargs)

        return self._fill(__fill, **kw_args)

    def power(self, a: Number, size: Union[int, tuple, None] = None) -> Union[np.ndarray, Number]:
        """Draw from a power distribution.

        Args:
            a: Parameter of the distribution. Must be non-negative.
            size: Output shape.

        Returns:
            numpy.ndarray or scalar.
        """
        self._check_shape(size)
        kw_args = {'a': a}

        if size is None:
            return self._random_generators[0].power(**kw_args)

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self._shp_max +
                (slice(first, last),)] = random_state.power(size=self._get_slice_size(first, last), **kwargs)

        return self._fill(__fill, **kw_args)

    def random(self, size: Union[int, tuple, None] = None, dtype=np.float64) -> Union[np.ndarray, Number]:
        """Return random floats in the half-open interval [0.0, 1.0).

        Args:
            size: Output shape.
            dtype: Dtype of output array.

        Returns:
            numpy.ndarray or scalar.
        """
        self._check_shape(size)
        kw_args = {'dtype': dtype}

        if size is None:
            return self._random_generators[0].random(**kw_args)

        if self._values.dtype != dtype:
            self._values = self._values.astype(dtype)

        def __fill(random_state, out, first, last, **kwargs):
            if self._shp_max == 0:
                random_state.random(out=out[(slice(None),) * self._shp_max + (slice(first, last),)], **kwargs)
            else:
                out[(slice(None),) * self._shp_max +
                    (slice(first, last),)] = random_state.random(size=self._get_slice_size(first, last), **kwargs)

        return self._fill(__fill, **kw_args)

    def rayleigh(self, scale: Number = 1.0, size: Union[int, tuple, None] = None) -> Union[np.ndarray, Number]:
        """Draw from a Rayleigh distribution.

        Args:
            scale: Scale, also equals the mode. Must be non-negative. Default is 1.
            size: Output shape.

        Returns:
            numpy.ndarray or scalar.
        """
        self._check_shape(size)
        kw_args = {'scale': scale}

        if size is None:
            return self._random_generators[0].rayleigh(**kw_args)

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self._shp_max +
                (slice(first, last),)] = random_state.rayleigh(size=self._get_slice_size(first, last), **kwargs)

        return self._fill(__fill, **kw_args)

    def standard_cauchy(self, size: Union[int, tuple, None] = None) -> Union[np.ndarray, Number]:
        """Draw from a standard Cauchy distribution.

        Args:
            size (int or tuple): Output shape.

        Returns:
            numpy.ndarray or scalar.
        """
        self._check_shape(size)
        kw_args = {}

        if size is None:
            return self._random_generators[0].standard_cauchy(**kw_args)

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self._shp_max +
                (slice(first, last),)] = random_state.standard_cauchy(size=self._get_slice_size(first, last),
                                                                      **kwargs)

        return self._fill(__fill, **kw_args)

    def standard_exponential(self, size: Union[int, tuple, None] = None, dtype=np.float64, method: str = 'zig') -> Union[np.ndarray, Number]:
        """Draw from a standard exponential distribution.

        Args:
            size: Output shape.
            dtype: Dtype of output array.
            method: Either ‘inv’ or ‘zig’. ‘inv’ uses the inverse CDF method. ‘zig’ uses the much faster Ziggurat
                method of Marsaglia and Tsang.

        Returns:
            numpy.ndarray or scalar.
        """
        self._check_shape(size)
        kw_args = {'dtype': dtype,
                   'method': method}

        if size is None:
            return self._random_generators[0].standard_exponential(**kw_args)

        if self._values.dtype != dtype:
            self._values = self._values.astype(dtype)

        def __fill(random_state, out, first, last, **kwargs):
            if self._shp_max == 0:
                random_state.standard_exponential(out=out[(slice(None),) * self._shp_max + (slice(first, last),)],
                                                  **kwargs)
            else:
                out[(slice(None),) * self._shp_max +
                    (slice(first, last),)] = random_state.standard_exponential(size=self._get_slice_size(first, last),
                                                                               **kwargs)

        return self._fill(__fill, **kw_args)

    def standard_gamma(self, shape: Number, size: Union[int, tuple, None] = None, dtype=np.float64) -> Union[np.ndarray, Number]:
        """Draw from a standard gamma distribution.

        Args:
            shape: Parameter, must be non-negative.
            size: Output shape.
            dtype: Dtype of output array.

        Returns:
            numpy.ndarray or scalar.
        """
        self._check_shape(size)
        kw_args = {'shape': shape,
                   'dtype': dtype}

        if size is None:
            return self._random_generators[0].standard_gamma(**kw_args)

        if self._values.dtype != dtype:
            self._values = self._values.astype(dtype)

        def __fill(random_state, out, first, last, **kwargs):
            if self._shp_max == 0:
                random_state.standard_gamma(out=out[(slice(None),) * self._shp_max + (slice(first, last),)],
                                            **kwargs)
            else:
                out[(slice(None),) * self._shp_max +
                    (slice(first, last),)] = random_state.standard_gamma(size=self._get_slice_size(first, last),
                                                                         **kwargs)

        return self._fill(__fill, **kw_args)

    def standard_normal(self, size: Union[int, tuple, None] = None, dtype=np.float64) -> Union[np.ndarray, Number]:
        """Draw from a standard normal distribution.

        Args:
            size: Output shape.
            dtype: Dtype of output array.

        Returns:
            numpy.ndarray or scalar.
        """
        self._check_shape(size)
        kw_args = {'dtype': dtype}

        if size is None:
            return self._random_generators[0].standard_normal(**kw_args)

        if self._values.dtype != dtype:
            self._values = self._values.astype(dtype)

        def __fill(random_state, out, first, last, **kwargs):
            if self._shp_max == 0:
                random_state.standard_normal(out=out[(slice(None),) * self._shp_max + (slice(first, last),)],
                                             **kwargs)
            else:
                out[(slice(None),) * self._shp_max +
                    (slice(first, last),)] = random_state.standard_normal(size=self._get_slice_size(first, last),
                                                                          **kwargs)

        return self._fill(__fill, **kw_args)

    def standard_t(self, df: Number, size: Union[int, tuple, None] = None) -> Union[np.ndarray, Number]:
        """Draw from a standard Student’s t distribution.

        Args:
            df: Degrees of freedom, must be > 0.
            size: Output shape.

        Returns:
            numpy.ndarray or scalar.
        """
        self._check_shape(size)
        kw_args = {'df': df}

        if size is None:
            return self._random_generators[0].standard_t(**kw_args)

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self._shp_max +
                (slice(first, last),)] = random_state.standard_t(size=self._get_slice_size(first, last),
                                                                 **kwargs)

        return self._fill(__fill, **kw_args)

    def triangular(self, left: Number, mode: Number, right: Number, size: Union[int, tuple, None] = None) -> Union[np.ndarray, Number]:
        """Draw from a triangular distribution.

        Args:
            left: Lower limit.
            mode: The value where the peak of the distribution occurs. The value must fulfill the condition
                left <= mode <= right.
            right: Upper limit, must be larger than left.
            size: Output shape.

        Returns:
            numpy.ndarray or scalar.
        """
        self._check_shape(size)
        kw_args = {'left': left,
                   'mode': mode,
                   'right': right}

        if size is None:
            return self._random_generators[0].triangular(**kw_args)

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self._shp_max +
                (slice(first, last),)] = random_state.triangular(size=self._get_slice_size(first, last), **kwargs)

        return self._fill(__fill, **kw_args)

    def uniform(self, low: Number = 0.0, high: Number = 1.0, size: Union[int, tuple, None] = None) -> Union[np.ndarray, Number]:
        """Draw from a uniform distribution.

        Args:
            low: Lower bound.
            high: Upper bound.
            size: Output shape.

        Returns:
            numpy.ndarray or scalar.
        """
        self._check_shape(size)
        kw_args = {'low': low,
                   'high': high}

        if size is None:
            return self._random_generators[0].uniform(**kw_args)

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self._shp_max +
                (slice(first, last),)] = random_state.uniform(size=self._get_slice_size(first, last), **kwargs)

        return self._fill(__fill, **kw_args)

    def vonmises(self, mu: Number, kappa: Number, size: Union[int, tuple, None] = None) -> Union[np.ndarray, Number]:
        """Draw from a von Mises distribution.

        Args:
            mu: Mode (“center”) of the distribution.
            kappa: Dispersion of the distribution, has to be >=0.
            size: Output shape.

        Returns:
            numpy.ndarray or scalar.
        """
        self._check_shape(size)
        kw_args = {'mu': mu,
                   'kappa': kappa}

        if size is None:
            return self._random_generators[0].vonmises(**kw_args)

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self._shp_max +
                (slice(first, last),)] = random_state.vonmises(size=self._get_slice_size(first, last), **kwargs)

        return self._fill(__fill, **kw_args)

    def wald(self, mean: Number, scale: Number, size: Union[int, tuple, None] = None) -> Union[np.ndarray, Number]:
        """Draw from a Wald distribution.

        Args:
            mean: Distribution mean, must be > 0.
            scale: Scale parameter, must be > 0.
            size: Output shape.

        Returns:
            numpy.ndarray or scalar.
        """
        self._check_shape(size)
        kw_args = {'mean': mean,
                   'scale': scale}

        if size is None:
            return self._random_generators[0].wald(**kw_args)

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self._shp_max +
                (slice(first, last),)] = random_state.wald(size=self._get_slice_size(first, last), **kwargs)

        return self._fill(__fill, **kw_args)

    def weibull(self, a: Number, size: Union[int, tuple, None] = None) -> Union[np.ndarray, Number]:
        """Draw from a Weibull distribution.

        Args:
            a: Shape parameter of the distribution. Must be nonnegative.
            size: Output shape.

        Returns:
            numpy.ndarray or scalar.
        """
        self._check_shape(size)
        kw_args = {'a': a}

        if size is None:
            return self._random_generators[0].weibull(**kw_args)

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self._shp_max +
                (slice(first, last),)] = random_state.weibull(size=self._get_slice_size(first, last), **kwargs)

        return self._fill(__fill, **kw_args)

    def zipf(self, a: Number, size: Union[int, tuple, None] = None) -> Union[np.ndarray, Number]:
        """Draw from a Zipf distribution.

        Args:
            a: Distribution parameter. Must be greater than 1.
            size: Output shape.

        Returns:
            numpy.ndarray or scalar.
        """
        self._check_shape(size)
        kw_args = {'a': a}

        if size is None:
            return self._random_generators[0].zipf(**kw_args)

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self._shp_max +
                (slice(first, last),)] = random_state.zipf(size=self._get_slice_size(first, last), **kwargs)

        return self._fill(__fill, **kw_args)

    def _fill(self, func, **kwargs):
        """Send jobs to the threads."""
        with concurrent.futures.ThreadPoolExecutor(self._num_threads) as executor:
            futures = [executor.submit(func,
                                       self._random_generators[i],
                                       self._values,
                                       self._steps[i][0],
                                       self._steps[i][1],
                                       **kwargs)
                       for i in range(self._num_threads)]
            for fut in concurrent.futures.as_completed(futures):
                fut.result()

        values = self._values
        self._values = None
        return values

    def _check_shape(self, size):
        """Standard size checks to be done before execution of any distribution sampling."""
        if size != self._shape:
            if isinstance(size, (int, float, complex, np.integer, np.floating)):
                size = size,
            self._shape = size
            self._shp_max = argmax(self._shape) if size is not None else None
            self._steps = [(t * (self._shape[self._shp_max] // self._num_threads),
                            (t + 1) * (self._shape[self._shp_max] // self._num_threads))
                           if t < (self._num_threads - 1) else
                           (t * (self._shape[self._shp_max] // self._num_threads), self._shape[self._shp_max])
                           for t in range(self._num_threads)] if size is not None else []
        self._values = np.empty(self._shape) if size is not None else None

    def _get_slice_size(self, fst, lst):
        """Get the shape of the slice to be filled."""
        return tuple(x if i != self._shp_max else lst - fst for i, x in enumerate(self._shape))
