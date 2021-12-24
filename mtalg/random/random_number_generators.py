from numpy.random import default_rng, SeedSequence
import concurrent.futures
import numpy as np
from multiprocessing import cpu_count
from mtalg.tools.__check_threads import check_threads

argmax = lambda iterable: max(enumerate(iterable), key=lambda x: x[1])[0]


class MultithreadedRNG:
    """Multithreaded random number generator

    Args
        seed        (int): Random seed
        num_threads (int): Number of threads to be used
    """

    def __init__(self, seed=None, num_threads=None):
        self.num_threads = check_threads(num_threads or cpu_count())
        seq = SeedSequence(seed)
        self._random_generators = [default_rng(s) for s in seq.spawn(self.num_threads)]
        self.shape = 0,
        self.shp_max = 0
        self.values = np.empty(self.shape)
        self.steps = []

    def beta(self, size, a, b):
        """Draw from a beta distribution

        Args
            size (int or tuple): Output shape
            a           (float): Alpha, positive (>0)
            b           (float): Beta, positive (>0)
        """
        self._check_shape(size)
        kw_args = {'a': a,
                   'b': b}

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self.shp_max +
                (slice(first, last),)] = random_state.beta(size=self._get_slice_size(first, last), **kwargs)

        self._fill(__fill, **kw_args)

    def binomial(self, size, n, p):
        """Draw from a binomial distribution

        Args
            size (int or tuple): Output shape
            n           (float): Parameter of the distribution, >= 0. Floats are also accepted, but they will be
                                 truncated to integers
            p           (float): Parameter of the distribution, >= 0 and <=1
        """
        self._check_shape(size)
        kw_args = {'n': n,
                   'p': p}

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self.shp_max +
                (slice(first, last),)] = random_state.binomial(size=self._get_slice_size(first, last), **kwargs)

        self._fill(__fill, **kw_args)

    def chisquare(self, size, df):
        """Draw from a chisquare distribution

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

    def exponential(self, size, scale=1.0):
        """Draw from a exponential distribution

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

    def f(self, size, dfnum, dfden):
        """Draw from an F distribution

        Args
            size (int or tuple): Output shape
            dfnum       (float): Degrees of freedom in numerator, must be > 0
            dfden       (float): Degrees of freedom in denominator, must be > 0
        """
        self._check_shape(size)
        kw_args = {'dfnum': dfnum,
                   'dfden': dfden}

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self.shp_max +
                (slice(first, last),)] = random_state.f(size=self._get_slice_size(first, last), **kwargs)

        self._fill(__fill, **kw_args)

    def gamma(self, size, shape, scale=1.0):
        """Draw from a gamma distribution

        Args
            size (int or tuple): Output shape
            shape       (float): The shape of the gamma distribution. Must be non-negative
            scale       (float): The scale of the gamma distribution. Must be non-negative. Default is equal to 1
        """
        self._check_shape(size)
        kw_args = {'shape': shape,
                   'scale': scale}

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self.shp_max +
                (slice(first, last),)] = random_state.gamma(size=self._get_slice_size(first, last), **kwargs)

        self._fill(__fill, **kw_args)

    def geometric(self, size, p):
        """Draw from a geomatric distribution

        Args
            size (int or tuple): Output shape
            p           (float): The probability of success of an individual trial
        """
        self._check_shape(size)
        kw_args = {'p': p}

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self.shp_max +
                (slice(first, last),)] = random_state.geometric(size=self._get_slice_size(first, last), **kwargs)

        self._fill(__fill, **kw_args)

    def gumbel(self, size, loc=0.0, scale=1.0):
        """Draw from a Gumbel distribution

        Args
            size (int or tuple): Output shape
            loc         (float): The location of the mode of the distribution. Default is 0
            scale       (float): The scale parameter of the distribution. Default is 1. Must be non- negative
        """
        self._check_shape(size)
        kw_args = {'loc': loc,
                   'scale': scale}

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self.shp_max +
                (slice(first, last),)] = random_state.gumbel(size=self._get_slice_size(first, last), **kwargs)

        self._fill(__fill, **kw_args)

    def hypergeometric(self, size, ngood, nbad, nsample):
        """Draw from a hypergeometric distribution

        Args
            size (int or tuple): Output shape
            ngood         (int): Number of ways to make a good selection. Must be nonnegative and less than 10**9
            nbad          (int): Number of ways to make a bad selection. Must be nonnegative and less than 10**9
            nsample       (int): Number of items sampled. Must be nonnegative and less than ngood + nbad
        """
        self._check_shape(size)
        kw_args = {'ngood': ngood,
                   'nbad': nbad,
                   'nsample': nsample}

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self.shp_max +
                (slice(first, last),)] = random_state.hypergeometric(size=self._get_slice_size(first, last), **kwargs)

        self._fill(__fill, **kw_args)

    def laplace(self, size, loc=0.0, scale=1.0):
        """Draw from a Laplace distribution

        Args
            size (int or tuple): Output shape
            loc         (float): The position, μ, of the distribution peak. Default is 0
            scale       (float): λ, the exponential decay. Default is 1. Must be non- negative
        """
        self._check_shape(size)
        kw_args = {'loc': loc,
                   'scale': scale}

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self.shp_max +
                (slice(first, last),)] = random_state.laplace(size=self._get_slice_size(first, last),
                                                              **kwargs)

        self._fill(__fill, **kw_args)

    def logistic(self, size, loc=0.0, scale=1.0):
        """Draw from a logistic distribution

        Args
            size (int or tuple): Output shape
            loc         (float): Parameter of the distribution. Default is 0
            scale       (float): Parameter of the distribution. Must be non-negative. Default is 1
        """
        self._check_shape(size)
        kw_args = {'loc': loc,
                   'scale': scale}

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self.shp_max +
                (slice(first, last),)] = random_state.logistic(size=self._get_slice_size(first, last),
                                                               **kwargs)

        self._fill(__fill, **kw_args)

    def lognormal(self, size, mean=0.0, sigma=1.0):
        """Draw from a lognormal distribution

        Args
            size (int or tuple): Output shape
            mean        (float): Mean value of the underlying normal distribution. Default is 0
            sigma       (float): Standard deviation of the underlying normal distribution. Must be non-negative.
                                 Default is 1
        """
        self._check_shape(size)
        kw_args = {'mean': mean,
                   'sigma': sigma}

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self.shp_max +
                (slice(first, last),)] = random_state.lognormal(size=self._get_slice_size(first, last),
                                                                **kwargs)

        self._fill(__fill, **kw_args)

    def logseries(self, size, p):
        """Draw from a logarithmic series distribution

        Args
            size (int or tuple): Output shape
            p           (float): Shape parameter for the distribution. Must be in the range (0, 1)
        """
        self._check_shape(size)
        kw_args = {'p': p}

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self.shp_max +
                (slice(first, last),)] = random_state.logseries(size=self._get_slice_size(first, last), **kwargs)

        self._fill(__fill, **kw_args)

    def negative_binomial(self, size, n, p):
        """Draw from a negative binomial distribution

        Args
            size (int or tuple): Output shape
            n           (float): Parameter of the distribution, > 0
            p           (float): Parameter of the distribution. Must satisfy 0 < p <= 1
        """
        self._check_shape(size)
        kw_args = {'n': n,
                   'p': p}

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self.shp_max +
                (slice(first, last),)] = random_state.negative_binomial(size=self._get_slice_size(first, last),
                                                                        **kwargs)

        self._fill(__fill, **kw_args)

    def noncentral_chisquare(self, size, df, nonc):
        """Draw from a noncentral chisquare distribution

        Args
            size (int or tuple): Output shape
            df          (float): Number of degrees of freedom, must be > 0
            nonc        (float): Non-centrality, must be non-negative
        """
        self._check_shape(size)
        kw_args = {'df': df,
                   'nonc': nonc}

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self.shp_max +
                (slice(first, last),)] = random_state.noncentral_chisquare(size=self._get_slice_size(first, last),
                                                                           **kwargs)

        self._fill(__fill, **kw_args)

    def noncentral_f(self, size, dfnum, dfden, nonc):
        """Draw from a noncentral F distribution

        Args
            size (int or tuple): Output shape
            dfnum       (float): Degrees of freedom in numerator, must be > 0
            dfden       (float): Degrees of freedom in denominator, must be > 0
            nonc        (float): Non-centrality parameter, the sum of the squares of the numerator means, must be >= 0
        """
        self._check_shape(size)
        kw_args = {'dfnum': dfnum,
                   'dfden': dfden,
                   'nonc': nonc}

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self.shp_max +
                (slice(first, last),)] = random_state.noncentral_f(size=self._get_slice_size(first, last), **kwargs)

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

    def pareto(self, size, a):
        """Draw from a Pareto II or Lomax distribution

        Args
            size (int or tuple): Output shape
            a           (float): Shape of the distribution. Must be positive
        """
        self._check_shape(size)
        kw_args = {'a': a}

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self.shp_max +
                (slice(first, last),)] = random_state.pareto(size=self._get_slice_size(first, last), **kwargs)

        self._fill(__fill, **kw_args)

    def poisson(self, size, lam=1.0):
        """Draw from a poisson distribution

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

    def power(self, size, a):
        """Draw from a power distribution

        Args
            size (int or tuple): Output shape
            a           (float): Parameter of the distribution. Must be non-negative
        """
        self._check_shape(size)
        kw_args = {'a': a}

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self.shp_max +
                (slice(first, last),)] = random_state.power(size=self._get_slice_size(first, last), **kwargs)

        self._fill(__fill, **kw_args)

    def rayleigh(self, size, scale=1.0):
        """Draw from a Rayleigh distribution

        Args
            size (int or tuple): Output shape
            scale       (float): Scale, also equals the mode. Must be non-negative. Default is 1
        """
        self._check_shape(size)
        kw_args = {'scale': scale}

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self.shp_max +
                (slice(first, last),)] = random_state.rayleigh(size=self._get_slice_size(first, last), **kwargs)

        self._fill(__fill, **kw_args)

    def standard_cauchy(self, size):
        """Draw from a standard Cauchy distribution

        Args
            size (int or tuple): Output shape
        """
        self._check_shape(size)
        kw_args = {}

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self.shp_max +
                (slice(first, last),)] = random_state.standard_cauchy(size=self._get_slice_size(first, last),
                                                                      **kwargs)

        self._fill(__fill, **kw_args)

    def standard_exponential(self, size, dtype=np.float64, method='zig'):
        """Draw from a standard exponential distribution

        Args
            size (int or tuple): Output shape
            dtype       (dtype): Dtype of output array
            method        (str): Either ‘inv’ or ‘zig’. ‘inv’ uses the inverse CDF method. ‘zig’ uses the much
                                 faster Ziggurat method of Marsaglia and Tsang
        """
        self._check_shape(size)
        kw_args = {'dtype': dtype,
                   'method': method}
        if self.values.dtype != dtype:
            self.values = self.values.astype(dtype)

        def __fill(random_state, out, first, last, **kwargs):
            if self.shp_max == 0:
                random_state.standard_exponential(out=out[(slice(None),) * self.shp_max + (slice(first, last),)],
                                                  **kwargs)
            else:
                out[(slice(None),) * self.shp_max +
                    (slice(first, last),)] = random_state.standard_exponential(size=self._get_slice_size(first, last),
                                                                               **kwargs)

        self._fill(__fill, **kw_args)

    def standard_gamma(self, size, shape, dtype=np.float64):
        """Draw from a standard gamma distribution

        Args
            size (int or tuple): Output shape
            shap      e (float): Parameter, must be non-negative
            dtype       (dtype): Dtype of output array
        """
        self._check_shape(size)
        kw_args = {'shape': shape,
                   'dtype': dtype}
        if self.values.dtype != dtype:
            self.values = self.values.astype(dtype)

        def __fill(random_state, out, first, last, **kwargs):
            if self.shp_max == 0:
                random_state.standard_gamma(out=out[(slice(None),) * self.shp_max + (slice(first, last),)],
                                            **kwargs)
            else:
                out[(slice(None),) * self.shp_max +
                    (slice(first, last),)] = random_state.standard_gamma(size=self._get_slice_size(first, last),
                                                                         **kwargs)

        self._fill(__fill, **kw_args)

    def standard_normal(self, size, dtype=np.float64):
        """Draw from a standard normal distribution

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

    def standard_t(self, size, df):
        """Draw from a standard Student’s t distribution

        Args
            size (int or tuple): Output shape
            df          (float): Degrees of freedom, must be > 0
        """
        self._check_shape(size)
        kw_args = {'df': df}

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self.shp_max +
                (slice(first, last),)] = random_state.standard_t(size=self._get_slice_size(first, last),
                                                                 **kwargs)

        self._fill(__fill, **kw_args)

    def triangular(self, size, left, mode, right):
        """Draw from a triangular distribution

        Args
            size (int or tuple): Output shape
            left        (float): Lower limit
            mode        (float): The value where the peak of the distribution occurs. The value must fulfill the
                                 condition left <= mode <= right
            right       (float): Upper limit, must be larger than left
        """
        self._check_shape(size)
        kw_args = {'left': left,
                   'mode': mode,
                   'right': right}

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self.shp_max +
                (slice(first, last),)] = random_state.triangular(size=self._get_slice_size(first, last), **kwargs)

        self._fill(__fill, **kw_args)

    def uniform(self, size, low=0.0, high=1.0):
        """Draw from a uniform distribution

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

    def vonmises(self, size, mu, kappa):
        """Draw from a von Mises distribution

        Args
            size (int or tuple): Output shape
            mu          (float): Mode (“center”) of the distribution
            kappa       (float): Dispersion of the distribution, has to be >=0
        """
        self._check_shape(size)
        kw_args = {'mu': mu,
                   'kappa': kappa}

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self.shp_max +
                (slice(first, last),)] = random_state.vonmises(size=self._get_slice_size(first, last), **kwargs)

        self._fill(__fill, **kw_args)

    def wald(self, size, mean, scale):
        """Draw from a Wald distribution

        Args
            size (int or tuple): Output shape
            mean        (float): Distribution mean, must be > 0
            scale       (float): Scale parameter, must be > 0
        """
        self._check_shape(size)
        kw_args = {'mean': mean,
                   'scale': scale}

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self.shp_max +
                (slice(first, last),)] = random_state.wald(size=self._get_slice_size(first, last), **kwargs)

        self._fill(__fill, **kw_args)

    def weibull(self, size, a):
        """Draw from a Weibull distribution

        Args
            size (int or tuple): Output shape
            a           (float): Shape parameter of the distribution. Must be nonnegative
        """
        self._check_shape(size)
        kw_args = {'a': a}

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self.shp_max +
                (slice(first, last),)] = random_state.weibull(size=self._get_slice_size(first, last), **kwargs)

        self._fill(__fill, **kw_args)

    def zipf(self, size, a):
        """Draw from a Zipf distribution

        Args
            size (int or tuple): Output shape
            a           (float): Distribution parameter. Must be greater than 1
        """
        self._check_shape(size)
        kw_args = {'a': a}

        def __fill(random_state, out, first, last, **kwargs):
            out[(slice(None),) * self.shp_max +
                (slice(first, last),)] = random_state.zipf(size=self._get_slice_size(first, last), **kwargs)

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
