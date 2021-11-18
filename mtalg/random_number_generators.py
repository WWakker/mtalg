"""  Created on 15/11/2021::
------------- rng  -------------
**Authors**: W. Wakker

"""
from numpy.random import default_rng, SeedSequence
import multiprocessing
import concurrent.futures
import numpy as np


class MultithreadedRNG2D:
    """Generate random numbers from a standard normal distibution in a parallelized way for 2D arrays
    Currently only optimized for 2D arrays with more rows than columns

    Args
        shape         (tuple): (number of rows, number of columns)
        seed    (None or int): Seed for RNG
        threads (None or int): Number of threads
    """
    def __init__(self, shape, seed=None, threads=None):

        self.threads = threads or multiprocessing.cpu_count()

        seq = SeedSequence(seed)
        self._random_generators = [default_rng(s)
                                   for s in seq.spawn(threads)]

        self.shape = shape
        self.executor = concurrent.futures.ThreadPoolExecutor(threads)
        self.values = np.empty(shape)
        self.steps = [(t * (shape[0] // threads), (t + 1) * (shape[0] // threads))
                      if t < (threads - 1)
                      else (t * (shape[0] // threads), shape[0])
                      for t in range(threads)]

    def fill(self):
        def _fill(random_state, out, firstrow, lastrow):
            random_state.standard_normal(out=out[firstrow:lastrow])

        futures = {}
        for i in range(self.threads):
            args = (_fill,
                    self._random_generators[i],
                    self.values,
                    self.steps[i][0],
                    self.steps[i][1])
            futures[self.executor.submit(*args)] = i
        concurrent.futures.wait(futures)

    def __del__(self):
        self.executor.shutdown(False)
