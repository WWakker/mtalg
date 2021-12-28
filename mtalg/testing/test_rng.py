import mtalg
from mtalg.random import MultithreadedRNG
import numpy as np


class TestMRNG:
    def test1(self):
        mrng = MultithreadedRNG(seed=1)
        assert mrng._shape is None

    def test2(self):
        mrng = MultithreadedRNG(seed=1, num_threads=4)
        a = mrng.standard_normal(size=4)
        assert a.shape == (4,)

    def test3(self):
        mrng = MultithreadedRNG(seed=1, num_threads=4)
        a = mrng.standard_normal(4)
        b = mrng.standard_normal(4)
        assert (a != b).all()

    def test4(self):
        mrng = MultithreadedRNG(seed=1, num_threads=4)
        a = mrng.standard_normal((2, 2))
        assert a.shape == (2, 2)

    def test5(self):
        mrng = MultithreadedRNG(seed=1, num_threads=4)
        a = mrng.standard_normal(4)
        mrng = MultithreadedRNG(seed=2, num_threads=4)
        b = mrng.standard_normal(4)
        assert (a != b).all()

    def test6(self):
        mrng = MultithreadedRNG(seed=1, num_threads=4)
        a = mrng.standard_normal(4)
        mrng = MultithreadedRNG(seed=1, num_threads=4)
        b = mrng.standard_normal(4)
        assert (b == a).all()

    def test7(self):
        mrng = MultithreadedRNG(seed=1, num_threads=4)
        a = mrng.uniform(size=(10, 10), low=-1000, high=1000)
        assert not ((a > 0) & (a < 1)).all()

    def test8(self):
        mrng = MultithreadedRNG(seed=1, num_threads=4)
        a = mrng.uniform(size=(10, 10), low=0, high=1)
        assert ((a > 0) & (a < 1)).all()

    def test9(self):
        mrng = MultithreadedRNG(seed=1, num_threads=4)
        a = mrng.normal(size=(20, 10, 30, 100, 50), loc=1, scale=2)
        assert a.shape == (20, 10, 30, 100, 50)
        assert mrng._steps == [(0, 25), (25, 50), (50, 75), (75, 100)]
        assert np.isclose(np.std(a), 2, 1e-2)
        assert np.isclose(np.mean(a), 1, 1e-2)

    def test10(self):
        for gen in [np.random.PCG64, np.random.MT19937, np.random.Philox, np.random.SFC64]:
            mrng = MultithreadedRNG(seed=1, num_threads=4, bit_generator=gen)
            for size in [500, (10, 50), (10, 10, 50), (10, 50, 10)]:
                mrng.beta(size=size, a=0.01, b=0.01)
                mrng.binomial(size=size, n=10, p=0.5)
                mrng.chisquare(size=size, df=100)
                mrng.exponential(size=size)
                mrng.f(size=size, dfnum=1, dfden=1)
                mrng.gamma(size=size, shape=1)
                mrng.geometric(size=size, p=.5)
                mrng.gumbel(size=size, loc=1, scale=1)
                mrng.hypergeometric(size=size, ngood=10, nbad=10, nsample=20)
                mtalg.random.integers(size=size, low=0, high=10, endpoint=True)
                mtalg.random.integers(size=size, low=0, high=10, endpoint=True, dtype=np.int32)
                mrng.laplace(size=size, loc=1, scale=1)
                mrng.logistic(size=size, loc=1, scale=1)
                mrng.lognormal(size=size)
                mrng.logseries(size=size, p=.5)
                mrng.negative_binomial(size=size, n=10, p=0.5)
                mrng.noncentral_chisquare(size=size, df=10, nonc=1)
                mrng.noncentral_f(size=size, dfnum=1, dfden=1, nonc=1)
                mrng.normal(size=size, loc=1, scale=2)
                mrng.pareto(size=size, a=1)
                mrng.poisson(size=size)
                mrng.power(size=size, a=2)
                mrng.random(size=size)
                mrng.random(size=size, dtype=np.float32)
                mrng.rayleigh(size=size, scale=1)
                mrng.standard_cauchy(size=size)
                mrng.standard_exponential(size=size)
                mrng.standard_exponential(size=size, dtype=np.float32)
                mrng.standard_gamma(size=size, shape=1)
                mrng.standard_gamma(size=size, shape=1, dtype=np.float32)
                mrng.standard_normal(size=size)
                mrng.standard_normal(size=size, dtype=np.float32)
                mrng.standard_t(size=size, df=10)
                mrng.triangular(size=size, left=0, mode=5, right=10)
                mrng.uniform(size=size, low=0, high=10)
                mrng.vonmises(size=size, mu=0, kappa=1)
                mrng.wald(size=size, mean=1, scale=1)
                mrng.weibull(size=size, a=2)
                mrng.zipf(size=size, a=2)

    def test11(self):
        mrng = MultithreadedRNG(seed=1, num_threads=4)
        mrng.standard_normal(size=(20, 20, 1000))

    def test12(self):
        for gen in [np.random.PCG64, np.random.MT19937, np.random.Philox, np.random.SFC64]:
            mrng = MultithreadedRNG(seed=1, num_threads=4, bit_generator=gen)
            size = None
            a = mrng.beta(size=size, a=0.01, b=0.01)
            assert isinstance(a, float)
            mrng.binomial(size=size, n=10, p=0.5)
            mrng.chisquare(size=size, df=100)
            mrng.exponential(size=size)
            mrng.f(size=size, dfnum=1, dfden=1)
            mrng.gamma(size=size, shape=1)
            mrng.geometric(size=size, p=.5)
            mrng.gumbel(size=size, loc=1, scale=1)
            mrng.hypergeometric(size=size, ngood=10, nbad=10, nsample=20)
            mtalg.random.integers(size=size, low=0, high=10, endpoint=True)
            mtalg.random.integers(size=size, low=0, high=10, endpoint=True, dtype=np.int32)
            mrng.laplace(size=size, loc=1, scale=1)
            mrng.logistic(size=size, loc=1, scale=1)
            mrng.lognormal(size=size)
            mrng.logseries(size=size, p=.5)
            mrng.negative_binomial(size=size, n=10, p=0.5)
            mrng.noncentral_chisquare(size=size, df=10, nonc=1)
            mrng.noncentral_f(size=size, dfnum=1, dfden=1, nonc=1)
            mrng.normal(size=size, loc=1, scale=2)
            mrng.pareto(size=size, a=1)
            mrng.poisson(size=size)
            mrng.power(size=size, a=2)
            mrng.random(size=size)
            mrng.random(size=size, dtype=np.float32)
            mrng.rayleigh(size=size, scale=1)
            mrng.standard_cauchy(size=size)
            mrng.standard_exponential(size=size)
            mrng.standard_exponential(size=size, dtype=np.float32)
            mrng.standard_gamma(size=size, shape=1)
            mrng.standard_normal(size=size)
            mrng.standard_normal(size=size, dtype=np.float32)
            mrng.standard_t(size=size, df=10)
            mrng.triangular(size=size, left=0, mode=5, right=10)
            mrng.uniform(size=size, low=0, high=10)
            mrng.vonmises(size=size, mu=0, kappa=1)
            mrng.wald(size=size, mean=1, scale=1)
            mrng.weibull(size=size, a=2)
            mrng.zipf(size=size, a=2)

    def test13(self):
        mrng = MultithreadedRNG(num_threads=4)
        mtalg.set_num_threads(2)
        mrng2 = MultithreadedRNG()
        mrng3 = MultithreadedRNG(num_threads=3)
        assert mrng._num_threads == 4 and mrng2._num_threads == 2 and mrng3._num_threads == 3
