import concurrent.futures
import numpy as np
import mtalg.core.threads
from typing import Optional, Union
from numbers import Number

argmax = lambda iterable: max(enumerate(iterable), key=lambda x: x[1])[0]


def _add_inplace(x, y): x += y


def _sub_inplace(x, y): x -= y


def _mul_inplace(x, y): x *= y


def _div_inplace(x, y): x /= y


def _pow_inplace(x, y): x **= y


def add(a: Union[np.ndarray, Number], b: Union[np.ndarray, Number], num_threads: Optional[int] = None, direction: str = 'left'):
    """Add multithreaded.

    Modifies a or b in-place depending on the direction.

    Args:
        a: Numpy array or scalar.
        b: Numpy array or scalar.
        num_threads: Number of threads to be used, overrides threads as set by :func:`~mtalg.set_num_threads`.
        direction: 'left' or 'right' to decide if a or b is modified.

    Examples:
         Add b to a; a is modified in-place.

         >>> a = mtalg.random.standard_normal(size=(10_000, 5_000))
         >>> b = mtalg.random.uniform(size=(10_000, 5_000), low=0, high=10)
         >>> mtalg.add(a, b, direction='left')

         Add a to b; b is modified in-place.

         >>> mtalg.add(a, b, direction='right')

         Add a to b using 4 threads.

         >>> mtalg.add(a, b, num_threads=4, direction='right')
    """
    __multithreaded_opr_direction(a, b, _add_inplace, num_threads, direction=direction)


def sub(a: Union[np.ndarray, Number], b: Union[np.ndarray, Number], num_threads: Optional[int] = None, direction: str = 'left'):
    """Subtract multithreaded.

    Modifies a or b in-place depending on the direction.

    Args:
        a: Numpy array or scalar.
        b: Numpy array or scalar.
        num_threads: Number of threads to be used, overrides threads as set by :func:`~mtalg.set_num_threads`.
        direction: 'left' or 'right' to decide if a or b is modified.

    Examples:
         Subtract b from a; a is modified in-place.

         >>> a = mtalg.random.standard_normal(size=(10_000, 5_000))
         >>> b = mtalg.random.uniform(size=(10_000, 5_000), low=0, high=10)
         >>> mtalg.sub(a, b, direction='left')

         Subtract a from b; b is modified in-place.

         >>> mtalg.sub(a, b, direction='right')

         Subtract a from b using 4 threads.

         >>> mtalg.sub(a, b, num_threads=4, direction='right')
    """
    __multithreaded_opr_direction(a, b, _sub_inplace, num_threads, direction=direction)


def mul(a: Union[np.ndarray, Number], b: Union[np.ndarray, Number], num_threads: Optional[int] = None, direction: str = 'left'):
    """Multiply multithreaded.

    Modifies a or b in-place depending on the direction.

    Args:
        a: Numpy array or scalar.
        b: Numpy array or scalar.
        num_threads: Number of threads to be used, overrides threads as set by :func:`~mtalg.set_num_threads`.
        direction: 'left' or 'right' to decide if a or b is modified.

    Examples:
         Multiply a by b; a is modified in-place.

         >>> a = mtalg.random.standard_normal(size=(10_000, 5_000))
         >>> b = mtalg.random.uniform(size=(10_000, 5_000), low=0, high=10)
         >>> mtalg.mul(a, b, direction='left')

         Multiply b by a; b is modified in-place.

         >>> mtalg.mul(a, b, direction='right')

         Multiply b by a using 4 threads.

         >>> mtalg.mul(a, b, num_threads=4, direction='right')
    """
    __multithreaded_opr_direction(a, b, _mul_inplace, num_threads, direction=direction)


def div(a: Union[np.ndarray, Number], b: Union[np.ndarray, Number], num_threads: Optional[int] = None, direction: str = 'left'):
    """Divide multithreaded.

    Modifies a or b in-place depending on the direction.

    Args:
        a: Numpy array or scalar.
        b: Numpy array or scalar.
        num_threads: Number of threads to be used, overrides threads as set by :func:`~mtalg.set_num_threads`.
        direction: 'left' or 'right' to decide if a or b is modified.

    Examples:
         Divide a by b; a is modified in-place.

         >>> a = mtalg.random.standard_normal(size=(10_000, 5_000))
         >>> b = mtalg.random.uniform(size=(10_000, 5_000), low=0, high=10)
         >>> mtalg.div(a, b, direction='left')

         Divide b by a; b is modified in-place.

         >>> mtalg.div(a, b, direction='right')

         Divide b by a using 4 threads.

         >>> mtalg.div(a, b, num_threads=4, direction='right')
    """
    __multithreaded_opr_direction(a, b, _div_inplace, num_threads, direction=direction)


def pow(a: Union[np.ndarray, Number], b: Union[np.ndarray, Number], num_threads: Optional[int] = None, direction: str = 'left'):
    """Raise to power multithreaded.

    Modifies a or b in-place depending on the direction.

    Args:
        a: Numpy array or scalar.
        b: Numpy array or scalar.
        num_threads: Number of threads to be used, overrides threads as set by :func:`~mtalg.set_num_threads`.
        direction: 'left' or 'right' to decide if a or b is modified.

    Examples:
         Raise a to the bth power; a is modified in-place.

         >>> a = mtalg.random.standard_normal(size=(10_000, 5_000))
         >>> b = mtalg.random.uniform(size=(10_000, 5_000), low=0, high=10)
         >>> mtalg.pow(a, b, direction='left')

         Raise b to the ath power; b is modified in-place.

         >>> mtalg.pow(a, b, direction='right')

         Raise b to the ath power b using 4 threads.

         >>> mtalg.pow(a, b, num_threads=4, direction='right')
    """
    __multithreaded_opr_direction(a, b, _pow_inplace, num_threads, direction=direction)


def __multithreaded_opr_direction(a, b, opr, num_threads, direction='left'):
    if direction == 'left':
        __multithreaded_opr(a, b, opr, num_threads=num_threads)
    elif direction == 'right':
        __multithreaded_opr(b, a, opr, num_threads=num_threads)
    else:
        raise ValueError(f"Invalid direction {direction}. Must take value either 'left' or 'right'. ")


def __multithreaded_opr(a, b, opr, num_threads: int = None):
    """Modifies a in-place.

    Beats numpy from around 1e7 operations onwards.

    Args:
      a (numpy.array): Left array to be summed. Modified in place.
      b (numpy.array): Right array to be summed.
      num_threads (int or None): Number of threads.
    """
    a_scalar = isinstance(a, (int, float, complex, np.integer, np.floating))
    b_scalar = isinstance(b, (int, float, complex, np.integer, np.floating))

    if not a_scalar and not b_scalar:
        scalar = False
        assert a.shape == b.shape, 'Shapes of both arrays must be the same'
    else:
        scalar = True
        if a_scalar:
            raise ValueError('Cannot modify scalar in-place')

    shape = a.shape
    shp_max = argmax(shape)
    num_threads = num_threads or mtalg.core.threads._global_num_threads
    assert num_threads > 0 and isinstance(num_threads, int), \
        f'Number of threads must be an integer > 0, found: {num_threads}'
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


def std(a: np.ndarray) -> Number:
    """Numba version of np.std.

    Args:
        a: np.ndarray.

    Returns:
        standard deviation.
    """
    try:
        from numba import njit
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Optional dependency missing: 'numba'.\n"
                                  "Install via pip as `pip install numba` or via conda as `conda install numba`.")

    @njit(parallel=True)
    def std_numba(x):
        return np.std(x)

    return std_numba(a)
