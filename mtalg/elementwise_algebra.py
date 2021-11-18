import numpy as np
from numpy.random import default_rng
import multiprocessing
import concurrent.futures

argmax = lambda iterable: max(enumerate(iterable), key=lambda x: x[1])[0]


def _add_inplace(x,y): x += y
def _sub_inplace(x,y): x -= y
def _mul_inplace(x,y): x *= y
def _div_inplace(x,y): x /= y
def _pow_inplace(x,y): x **= y

def add_MultiThreaded(a, b, threads, direction='left'):
    __MultiThreaded_opr_direction(a, b, _add_inplace, threads, direction=direction)
def sub_MultiThreaded(a, b, threads, direction='left'):
    __MultiThreaded_opr_direction(a, b, _sub_inplace, threads, direction=direction)
def mul_MultiThreaded(a, b, threads, direction='left'):
    __MultiThreaded_opr_direction(a, b, _mul_inplace, threads, direction=direction)
def div_MultiThreaded(a, b, threads, direction='left'):
    __MultiThreaded_opr_direction(a, b, _div_inplace, threads, direction=direction)
def pow_MultiThreaded(a, b, threads, direction='left'):
    __MultiThreaded_opr_direction(a, b, _pow_inplace, threads, direction=direction)


def __MultiThreaded_opr_direction(a, b, opr, threads, direction='left'):
    if direction == 'left':
        __MultiThreaded_opr(a, b, opr, threads=threads)
    elif direction == 'right':
        __MultiThreaded_opr(b, a, opr, threads=threads)
    else:
        raise ValueError(f"Invalid direction {direction}. Must take value either 'left' or 'right'. ")

def __MultiThreaded_opr(a, b, opr, threads=None):
    """Modifies a inplace; beats numpy from around 1e7 operations onwards.
    Supports tensors form n=1 up to n=6.
    Args:
      a (numpy.array): Left array to be summed. Modified in place.
      b (numpy.array): Right array to be summed.
      threads (int or None): Number of threads.
    """
    assert a.shape == b.shape

    shape = a.shape
    shp_max = argmax(shape)

    threads = threads or multiprocessing.cpu_count()
    executor = concurrent.futures.ThreadPoolExecutor(threads)
    steps = [(t * (a.shape[shp_max] // threads), (t + 1) * (a.shape[shp_max] // threads))
             if t < (threads - 1) else
             (t * (a.shape[shp_max] // threads), a.shape[shp_max])
             for t in range(threads)]

    if shp_max==0:
        def _fill(firstrow, lastrow):
            opr(a[firstrow:lastrow], b[firstrow:lastrow])
    elif shp_max==1:
        def _fill(firstrow, lastrow):
            opr(a[:, firstrow:lastrow], b[:,firstrow:lastrow])
    elif shp_max==2:
        def _fill(firstrow, lastrow):
            opr(a[:,:, firstrow:lastrow],  b[:,:,firstrow:lastrow])
    elif shp_max==3:
        def _fill(firstrow, lastrow):
            opr(a[:,:,:, firstrow:lastrow],  b[:,:,:,firstrow:lastrow])
    elif shp_max==4:
        def _fill(firstrow, lastrow):
            opr(a[:,:,:,:, firstrow:lastrow],  b[:,:,:,:,firstrow:lastrow])
    elif shp_max==5:
        def _fill(firstrow, lastrow):
            opr(a[:,:,:,:,:, firstrow:lastrow],  b[:,:,:,:,:,firstrow:lastrow])

    futures = {}
    for i in range(threads):
        args = (_fill, steps[i][0], steps[i][1])
        futures[executor.submit(*args)] = i
    concurrent.futures.wait(futures)
    executor.shutdown(False)



if __name__=='__main__':
    rng = default_rng(1)
    a = rng.standard_normal((int(4e5), 10, 10, 2))
    b = rng.standard_normal((int(4e5), 10, 10, 2))
    b = b.clip(min=1e-5)
    aT, bT = a.T, b.T

    c=a+b
    add_MultiThreaded(a, b, threads=24)
    (c==a).all()


    %timeit a + b
    %timeit aT + bT
    %timeit add_MultiThreaded(a, b, threads=4)
    %timeit add_MultiThreaded(a, -b, threads=4)

    %timeit a + b
    %timeit add_MultiThreaded(a, b, threads=4)
    %timeit a - b
    %timeit sub_MultiThreaded(a, b, threads=4)
    %timeit a * b
    %timeit mul_MultiThreaded(a, b, threads=4)
    %timeit a / b
    %timeit div_MultiThreaded(a, b, threads=4)
    %timeit a ** b
    %timeit pow_MultiThreaded(a, b, threads=4)





    import os
    os.environ['NUMEXPR_NUM_THREADS'] = '4'
    import numexpr as ne
    %timeit ne.evaluate('a+b')



    from numba import jit
    @jit(parallel=True)
    def add_numba(a, b):
        return a + b

    from numpy.random import default_rng

    rng = default_rng(1)

    a = rng.standard_normal((int(4e6)))
    b = rng.standard_normal((int(4e6)))
    %timeit add_numba(a, b)