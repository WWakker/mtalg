from mtalg.random import MultithreadedRNG
import numpy as np


class TestMRNG:
    def test1(self):
        mrng = MultithreadedRNG(seed=1)
        assert mrng.values.size == 0

    def test2(self):
        mrng = MultithreadedRNG(seed=1, num_threads=4)
        mrng.standard_normal(4)
        assert mrng.values.shape == (4,)

    def test3(self):
        mrng = MultithreadedRNG(seed=1, num_threads=4)
        mrng.standard_normal(4)
        a = mrng.values
        mrng.standard_normal(4)
        assert (a != mrng.values).all()

    def test4(self):
        mrng = MultithreadedRNG(seed=1, num_threads=4)
        mrng.standard_normal((2, 2))
        assert mrng.values.shape == (2, 2)

    def test5(self):
        mrng = MultithreadedRNG(seed=1, num_threads=4)
        mrng.standard_normal(4)
        a = mrng.values
        mrng = MultithreadedRNG(seed=2, num_threads=4)
        mrng.standard_normal(4)
        assert (a != mrng.values).all()

    def test6(self):
        mrng = MultithreadedRNG(seed=1, num_threads=4)
        mrng.standard_normal(4)
        a = mrng.values
        mrng = MultithreadedRNG(seed=1, num_threads=4)
        mrng.standard_normal(4)
        assert (mrng.values == a).all()

    def test7(self):
        mrng = MultithreadedRNG(seed=1, num_threads=4)
        mrng.uniform(size=(10, 10), low=-1000, high=1000)
        assert not ((mrng.values > 0) & (mrng.values < 1)).all()

    def test8(self):
        mrng = MultithreadedRNG(seed=1, num_threads=4)
        mrng.uniform(size=(10, 10), low=0, high=1)
        assert ((mrng.values > 0) & (mrng.values < 1)).all()

    def test9(self):
        mrng = MultithreadedRNG(seed=1, num_threads=4)
        mrng.normal((20, 10, 30, 100, 50), loc=1, scale=2)
        assert mrng.shp_max == 3
        assert mrng.values.shape == (20, 10, 30, 100, 50)
        assert mrng.steps == [(0, 25), (25, 50), (50, 75), (75, 100)]
        assert np.isclose(np.std(mrng.values), 2, 1e-2)
        assert np.isclose(np.mean(mrng.values), 1, 1e-2)

    def test10(self):
        mrng = MultithreadedRNG(seed=1, num_threads=4)
        mrng.beta(size=500, a=0.01, b=0.01)
        mrng.binomial(size=500, n=10, p=0.5)
        mrng.chisquare(size=500, df=100)
        mrng.exponential(size=500)
        mrng.poisson(size=500)
        mrng.standard_normal(size=500, dtype=np.float32)

    def test11(self):
        mrng = MultithreadedRNG(seed=1, num_threads=4)
        mrng.standard_normal(size=(20, 20, 1000))
