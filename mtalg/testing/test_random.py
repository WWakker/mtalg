import mtalg
import numpy as np
import pytest


class TestRandom:
    def test1(self):
        a = mtalg.random.standard_normal()
        assert isinstance(a, float)

    def test2(self):
        with pytest.raises(TypeError):
            mtalg.random.standard_normal(1, np.float32, 3)

    def test3(self):
        with pytest.raises(TypeError):
            mtalg.random.standard_normal(size=1, dtype=np.float32, wrong=4)

    def test4(self):
        for size in [None, 500, (10, 50), (10, 10, 50), (10, 50, 10)]:
            mtalg.random.beta(size=size, a=0.01, b=0.01)
            mtalg.random.binomial(size=size, n=10, p=0.5)
            mtalg.random.chisquare(size=size, df=100)
            mtalg.random.exponential(size=size)
            mtalg.random.f(size=size, dfnum=1, dfden=1)
            mtalg.random.gamma(size=size, shape=1)
            mtalg.random.geometric(size=size, p=.5)
            mtalg.random.gumbel(size=size, loc=1, scale=1)
            mtalg.random.hypergeometric(size=size, ngood=10, nbad=10, nsample=20)
            mtalg.random.integers(size=size, low=0, high=10, endpoint=True)
            mtalg.random.integers(size=size, low=0, high=10, endpoint=True, dtype=np.int32)
            mtalg.random.laplace(size=size, loc=1, scale=1)
            mtalg.random.logistic(size=size, loc=1, scale=1)
            mtalg.random.lognormal(size=size)
            mtalg.random.logseries(size=size, p=.5)
            mtalg.random.negative_binomial(size=size, n=10, p=0.5)
            mtalg.random.noncentral_chisquare(size=size, df=10, nonc=1)
            mtalg.random.noncentral_f(size=size, dfnum=1, dfden=1, nonc=1)
            mtalg.random.normal(size=size, loc=1, scale=2)
            mtalg.random.pareto(size=size, a=1)
            mtalg.random.poisson(size=size)
            mtalg.random.power(size=size, a=2)
            mtalg.random.random(size=size)
            mtalg.random.random(size=size, dtype=np.float32)
            mtalg.random.rayleigh(size=size, scale=1)
            mtalg.random.standard_cauchy(size=size)
            mtalg.random.standard_exponential(size=size)
            mtalg.random.standard_exponential(size=size, dtype=np.float32)
            mtalg.random.standard_gamma(size=size, shape=1)
            mtalg.random.standard_gamma(size=size, shape=1, dtype=np.float32)
            mtalg.random.standard_normal(size=size)
            mtalg.random.standard_normal(size=size, dtype=np.float32)
            mtalg.random.standard_t(size=size, df=10)
            mtalg.random.triangular(size=size, left=0, mode=5, right=10)
            mtalg.random.uniform(size=size, low=0, high=10)
            mtalg.random.vonmises(size=size, mu=0, kappa=1)
            mtalg.random.wald(size=size, mean=1, scale=1)
            mtalg.random.weibull(size=size, a=2)
            mtalg.random.zipf(size=size, a=2)

    def test5(self):
        mtalg.set_num_threads(1)
        assert mtalg.random._RNG._num_threads == 1
        mtalg.set_num_threads(2)
        assert mtalg.random._RNG._num_threads == 2
