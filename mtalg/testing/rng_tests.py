from mtalg.random import MultithreadedRNG
import numpy as np

rng = MultithreadedRNG(seed=1)
assert rng.values.size == 0

rng = MultithreadedRNG(seed=1, num_threads=4)
rng.standard_normal(4)
assert rng.values.shape == (4,)

a = rng.values
rng.standard_normal(4)
assert (a != rng.values).all()

rng.standard_normal((2, 2))
assert rng.values.shape == (2, 2)

rng = MultithreadedRNG(seed=2, num_threads=4)
rng.standard_normal(4)
assert (a != rng.values).all()

rng = MultithreadedRNG(seed=1, num_threads=4)
rng.standard_normal(4)
assert (rng.values == a).all()
rng.uniform(size=(10, 10), low=-1000, high=1000)
assert not ((rng.values > 0) & (rng.values < 1)).all()

rng.uniform(size=(10, 10), low=0, high=1)
assert ((rng.values > 0) & (rng.values < 1)).all()

rng.normal((20, 10, 30, 100, 50), loc=1, scale=2)
assert rng.shp_max == 3
assert rng.values.shape == (20, 10, 30, 100, 50)
assert rng.steps == [(0, 25), (25, 50), (50, 75), (75, 100)]
assert np.isclose(np.std(rng.values), 2, 1e-2)
assert np.isclose(np.mean(rng.values), 1, 1e-2)

print('All RNG tests passed')
