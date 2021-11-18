from numpy.random import default_rng, SeedSequence
import multiprocessing
import concurrent.futures
import numpy as np
from mtalg.tools.__get_num_threads import MAX_NUM_THREADS
argmax = lambda iterable: max(enumerate(iterable), key=lambda x: x[1])[0]

NUM_THREADS = MAX_NUM_THREADS

class MultithreadedRNG:

    def __init__(self, seed=None, threads=None):

        self.threads = threads or max(threads, NUM_THREADS)
        seq = SeedSequence(seed)
        self._random_generators = [default_rng(s) for s in seq.spawn(threads)]
        self.shape = 0,
        self.shp_max = 0
        self.values = np.empty(self.shape)
        self.steps = []
        self.executor = concurrent.futures.ThreadPoolExecutor(threads)

    def standard_normal(self, size, dtype=np.float64):
        self._check_shape(size)
        kw_args = {'dtype': dtype}
        if self.values.dtype != dtype:
            self.values = self.values.astype(dtype)

        def __fill(random_state, out, first_row, last_row, **kwargs):
            random_state.standard_normal(out=out[(slice(None),) * self.shp_max + (slice(first_row, last_row),)],
                                         **kwargs)

        self._fill(__fill, **kw_args)

    def _fill(self, func, **kwargs):
        futures = {}
        for i in range(self.threads):
            args = (func,
                    self._random_generators[i],
                    self.values,
                    self.steps[i][0],
                    self.steps[i][1])
            futures[self.executor.submit(*args, **kwargs)] = i
        concurrent.futures.wait(futures)

    def _check_shape(self, size):
        if size != self.shape:
            self.shape = size
            self.shp_max = argmax(self.shape)
            self.values = np.empty(self.shape)
            self.steps = [(t * (self.shape[self.shp_max] // self.threads),
                           (t + 1) * (self.shape[self.shp_max] // self.threads))
                          if t < (self.threads - 1) else
                          (t * (self.shape[self.shp_max] // self.threads), self.shape[self.shp_max])
                          for t in range(self.threads)]

    def __del__(self):
        self.executor.shutdown(False)



if __name__=='__main__':
    pass