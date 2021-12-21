import multiprocessing
import concurrent.futures
from mtalg.tools.__get_num_threads import MAX_NUM_THREADS
import numpy as np

NUM_THREADS = MAX_NUM_THREADS

# TODO: doc string to be updated for all of them (ad hoc for each + one inherited from the function below?)
# TODO: check why nans in pow


argmax = lambda iterable: max(enumerate(iterable), key=lambda x: x[1])[0]


def _add_inplace(x, y): x += y


def _sub_inplace(x, y): x -= y


def _mul_inplace(x, y): x *= y


def _div_inplace(x, y): x /= y


def _pow_inplace(x, y): x **= y


def add(a, b, num_threads=NUM_THREADS, direction='left'):
    __MultiThreaded_opr_direction(a, b, _add_inplace, num_threads, direction=direction)


def sub(a, b, num_threads=NUM_THREADS, direction='left'):
    __MultiThreaded_opr_direction(a, b, _sub_inplace, num_threads, direction=direction)


def mul(a, b, num_threads=NUM_THREADS, direction='left'):
    __MultiThreaded_opr_direction(a, b, _mul_inplace, num_threads, direction=direction)


def div(a, b, num_threads=NUM_THREADS, direction='left'):
    __MultiThreaded_opr_direction(a, b, _div_inplace, num_threads, direction=direction)


def pow(a, b, num_threads=NUM_THREADS, direction='left'):
    __MultiThreaded_opr_direction(a, b, _pow_inplace, num_threads, direction=direction)


def __MultiThreaded_opr_direction(a, b, opr, num_threads, direction='left'):
    if direction == 'left':
        __MultiThreaded_opr(a, b, opr, num_threads=num_threads)
    elif direction == 'right':
        __MultiThreaded_opr(b, a, opr, num_threads=num_threads)
    else:
        raise ValueError(f"Invalid direction {direction}. Must take value either 'left' or 'right'. ")


def __MultiThreaded_opr(a, b, opr, num_threads=None):
    """Modifies a inplace; beats numpy from around 1e7 operations onwards.

    Args:
      a (numpy.array): Left array to be summed. Modified in place.
      b (numpy.array): Right array to be summed.
      num_threads (int or None): Number of num_threads.
    """
    scalar = True if isinstance(b, (int, float, complex)) and not isinstance(b, bool) else False
    if not scalar:
        assert a.shape == b.shape

    shape = a.shape
    shp_max = argmax(shape)
    num_threads = min(num_threads or float('inf'), NUM_THREADS)
    steps = [(t * (shape[shp_max] // num_threads), (t + 1) * (shape[shp_max] // num_threads))
             if t < (num_threads - 1) else
             (t * (shape[shp_max] // num_threads), shape[shp_max])
             for t in range(num_threads)]

    if scalar:
        def _fill(first, last):
            opr(a[(slice(None),) * shp_max + (slice(first, last),)], b)
    else:
        def _fill(first, last):
            opr(a[(slice(None),) * shp_max + (slice(first, last),)],
                b[(slice(None),) * shp_max + (slice(first, last),)])

    with concurrent.futures.ThreadPoolExecutor(num_threads) as executor:
        futures = [executor.submit(_fill, steps[i][0], steps[i][1]) for i in range(num_threads)]
        for fut in concurrent.futures.as_completed(futures):
            fut.result()


def std(a: np.ndarray):
    """Numba version of np.std

    Args
        a: np.ndarray

    Returns
        np.floating: standard deviation
    """
    try:
        from numba import njit
    except ImportError:
        raise ImportError("Optional dependency missing: 'numba'; Use pip or conda to install")

    @njit(parallel=True)
    def std_numba(x):
        return np.std(x)

    return std_numba(a)


if __name__ == '__main__':
    pass
