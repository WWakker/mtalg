from numpy.random import default_rng, SeedSequence
import multiprocessing
import concurrent.futures
import numpy as np
from mtalg.tools.__get_num_threads import MAX_NUM_THREADS
import os

argmax = lambda iterable: max(enumerate(iterable), key=lambda x: x[1])[0]
NUM_THREADS = MAX_NUM_THREADS


class MultithreadedRNG:
    """Multithreaded random number generator

    Args
        seed        (int): Random seed
        num_threads (int): Number of threads to be used
    """

    def __init__(self, seed=None, num_threads=None):
        self.num_threads = min(num_threads or float('inf'), NUM_THREADS)
        seq = SeedSequence(seed)
        self._random_generators = [default_rng(s) for s in seq.spawn(self.num_threads)]
        self.shape = 0,
        self.shp_max = 0
        self.values = np.empty(self.shape)
        self.steps = []

    def beta(self, size, a, b):
        """Draw from the beta distribution

        Args
            size (int or tuple): Output shape
            a           (float): alpha
            b           (float): beta
        """
        self._check_shape(size)
        kw_args = {'a': a,
                   'b': b}

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self.shp_max +
                (slice(first, last),)] = random_state.beta(size=self._get_slice_size(first, last), **kwargs)

        self._fill(__fill, **kw_args)

    def binomial(self, size, n, p):
        """Draw from the binomial distribution

        Args
            size (int or tuple): Output shape
            n           (float): n
            p           (float): p
        """
        self._check_shape(size)
        kw_args = {'n': n,
                   'p': p}

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self.shp_max +
                (slice(first, last),)] = random_state.binomial(size=self._get_slice_size(first, last), **kwargs)

        self._fill(__fill, **kw_args)

    def chisquare(self, size, df):
        """Draw from the chisquare distribution

        Args
            size (int or tuple): Output shape
            df          (float): Number of degrees of freedom, must be > 0
        """
        self._check_shape(size)
        kw_args = {'df': df}

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self.shp_max +
                (slice(first, last),)] = random_state.chisquare(size=self._get_slice_size(first, last), **kwargs)

        self._fill(__fill, **kw_args)

    def exponential(self, size, scale):
        """Draw from the exponential distribution

        Args
            size (int or tuple): Output shape
            scale       (float): The scale parameter, β = 1/λ Must be non-negative
        """
        self._check_shape(size)
        kw_args = {'scale': scale}

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self.shp_max +
                (slice(first, last),)] = random_state.exponential(size=self._get_slice_size(first, last), **kwargs)

        self._fill(__fill, **kw_args)

    def normal(self, size, loc=0.0, scale=1.0):
        """Draw from the normal distribution

        Args
            size (int or tuple): Output shape
            loc         (float): Mean of the distriution
            scale       (float): Standard deviation of the distriution
        """
        self._check_shape(size)
        kw_args = {'loc': loc,
                   'scale': scale}

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self.shp_max +
                (slice(first, last),)] = random_state.normal(size=self._get_slice_size(first, last), **kwargs)

        self._fill(__fill, **kw_args)

    def poisson(self, size, lam=1.0):
        """Draw from the poisson distribution

        Args
            size (int or tuple): Output shape
            lam         (float): Expected number of events occurring in a fixed-time interval, must be >= 0
        """
        self._check_shape(size)
        kw_args = {'lam': lam}

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self.shp_max +
                (slice(first, last),)] = random_state.poisson(size=self._get_slice_size(first, last), **kwargs)

        self._fill(__fill, **kw_args)

    def uniform(self, size, low=0.0, high=1.0):
        """Draw from the uniform distribution

        Args
            size (int or tuple): Output shape
            low         (float): Lower bound
            high        (float): Upper bound
        """
        self._check_shape(size)
        kw_args = {'low': low,
                   'high': high}

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self.shp_max +
                (slice(first, last),)] = random_state.uniform(size=self._get_slice_size(first, last), **kwargs)

        self._fill(__fill, **kw_args)

    def standard_normal(self, size, dtype=np.float64):
        """Draw from the standard normal distribution

        Args
            size (int or tuple): Output shape
            dtype       (dtype): Dtype of output array
        """
        self._check_shape(size)
        kw_args = {'dtype': dtype}
        if self.values.dtype != dtype:
            self.values = self.values.astype(dtype)

        def __fill(random_state, out, first, last, **kwargs):
            if self.shp_max == 0:
                random_state.standard_normal(out=out[(slice(None),) * self.shp_max + (slice(first, last),)],
                                             **kwargs)
            else:
                out[(slice(None),) * self.shp_max +
                    (slice(first, last),)] = random_state.standard_normal(size=self._get_slice_size(first, last),
                                                                          **kwargs)

        self._fill(__fill, **kw_args)

    def _fill(self, func, **kwargs):
        """Send jobs to the threads"""
        with concurrent.futures.ThreadPoolExecutor(self.num_threads) as executor:
            futures = [executor.submit(func, 
                                       self._random_generators[i],
                                       self.values, 
                                       self.steps[i][0],
                                       self.steps[i][1],
                                       **kwargs) 
                       for i in range(self.num_threads)]
            for fut in concurrent.futures.as_completed(futures):
                fut.result()

    def _check_shape(self, size):
        """Standard size checks to be done before execution of any distribution sampling"""
        if size != self.shape:
            if isinstance(size, (int, float, complex, np.integer, np.floating)):
                size = size,
            self.shape = size
            self.shp_max = argmax(self.shape)
            self.values = np.empty(self.shape)
            self.steps = [(t * (self.shape[self.shp_max] // self.num_threads),
                           (t + 1) * (self.shape[self.shp_max] // self.num_threads))
                          if t < (self.num_threads - 1) else
                          (t * (self.shape[self.shp_max] // self.num_threads), self.shape[self.shp_max])
                          for t in range(self.num_threads)]

    def _get_slice_size(self, fst, lst):
        """Get the shape of the slice to be filled"""
        return tuple(x if i != self.shp_max else lst - fst for i, x in enumerate(self.shape))


if __name__ == '__main__':
    pass
