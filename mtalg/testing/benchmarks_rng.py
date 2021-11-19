import numpy as np
from numpy.random import default_rng
import multiprocessing
import concurrent.futures
from numba import njit, prange, jit
import time

def add_threaded2D(a, b, threads=None):
    """Modifies a inplace; beats numpy from around 1e7 operations onwards.
    Args:
        a (numpy.array): Left array to be summed. Modified in place.
        b (numpy.array): Right array to be summed.
        threads (int or None): Number of threads.
    """
    assert a.shape == b.shape

    threads = threads or multiprocessing.cpu_count()
    executor = concurrent.futures.ThreadPoolExecutor(threads)
    steps = [(t * (a.shape[0] // threads), (t + 1) * (a.shape[0] // threads))
             if t < (threads - 1)
             else (t * (a.shape[0] // threads), a.shape[0])
             for t in range(threads)]

    def _fill(firstrow, lastrow):
        a[firstrow:lastrow] += b[firstrow:lastrow]

    futures = {}
    for i in range(threads):
        args = (_fill,
                steps[i][0],
                steps[i][1])
        futures[executor.submit(*args)] = i
    concurrent.futures.wait(futures)
    executor.shutdown(False)


rng = default_rng(1)
a = rng.standard_normal((int(1e6), 100))
b = rng.standard_normal((int(1e6), 100))

c = a + b
add_threaded2D(a, b, threads=24)
(c == a).all()

% timeit
a + b
% timeit
add_threaded2D(a, b, threads=24)


# Check against numba
@njit(parallel=True)
def f_add(a, b, inplace=True):
    if not inplace:
        return a + b
    a += b


c = a + b
f_add(a, b, inplace=True)
(c == a).all()

% timeit
a + b  # 2.67 s per loop
% timeit
add_threaded2D(a, b, threads=24)  # 131 ms per loop
% timeit
f_add(a, b, inplace=True)  # 778 ms per loop
% timeit
f_add(a, b, inplace=False)  # 638 ms per loop

c = a + b
d = f_add(a, b, inplace=False)
(c == d).all()


# Standard deviation
@njit(parallel=True)
def f_par_std(x):
    return np.std(x)


@njit
def f_std(x):
    return np.std(x)


np.isclose(a.std(), f_std(a))
np.isclose(a.std(), f_par_std(a))

% timeit
a.std()  # 3.61 s per loop
% timeit
f_std(a)  # 1.19 s per loop
% timeit
f_par_std(a)  # %timeit a.std()  # 95.6 ms per loop

### Errors

a = np.arange(4).reshape((2, 2))
print(a.dtype)  # int
b = np.ones((2, 2))
print(b.dtype)  # float

a[0] += b[
    0]  # UFuncTypeError: Cannot cast ufunc 'add' output from dtype('float64') to dtype('int64') with casting rule 'same_kind'
add_threaded2D(a, b, threads=2)  # No error


@njit(parallel=True)
def numba_add(a, b):
    n = a.shape[0]
    for i in prange(n):
        a[i] += b[i]


c = a + b
numba_add(a, b)


@njit(parallel=True, fastmath=True)
def numba_add(a, b):
    for i in prange(a.shape[0]):
        a[i] += b[i]


c = a + b
numba_add(a, b)
(c == a).all()

% time
add_threaded2D(a, b, threads=24)
% time
numba_add(a, b)


def naive_add(x, y):
    """
    add two arrays using a Python for-loop
    """
    z = np.empty_like(x)
    for i in range(len(x)):
        z[i] = x[i] + y[i]

    return z


@np.vectorize
def numpy_add_vectorize(x, y):
    return x + y


@jit  # this is the only thing we do different
def numba_add(x, y):
    """
    add two arrays using a Python for-loop
    """
    z = np.empty_like(x)
    for i in range(len(x)):
        z[i] = x[i] + y[i]

    return z


@jit(nopython=True, parallel=True)
def numba_add_parallel(x, y):
    """
    add two arrays using a Python for-loop
    """
    z = np.empty_like(x)
    for i in prange(len(x)):
        z[i] = x[i] + y[i]

    return z


# set up two vectors
n = 1_000_000
x = np.random.randn(n)
y = np.random.randn(n)

start = time.time()
z = naive_add(x, y)
end = time.time()
print("time for naive add: {:0.3e} sec".format(end - start))

start = time.time()
z = np.add(x, y)
end = time.time()
print("time for numpy add: {:0.3e} sec".format(end - start))

start = time.time()
z = numba_add(x, y)
end = time.time()
print("time for numba add: {:0.3e} sec".format(end - start))

start = time.time()
add_threaded2D(x, y)
end = time.time()
print("time for add threaded: {:0.3e} sec".format(end - start))

start = time.time()
z = numba_add_parallel(x, y)
end = time.time()
print("time for numba parallel add: {:0.3e} sec".format(end - start))

start = time.time()
z = np.add(x, y)
end = time.time()
print("time for numpy add: {:0.3e} sec".format(end - start))
