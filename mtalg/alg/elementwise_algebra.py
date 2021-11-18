import multiprocessing
import concurrent.futures
NUM_THREADS = multiprocessing.cpu_count()

# TODO: doc string to be updated for all of them (ad hoc for each + one inherited from the function below?)

argmax = lambda iterable: max(enumerate(iterable), key=lambda x: x[1])[0]


def _add_inplace(x,y): x += y
def _sub_inplace(x,y): x -= y
def _mul_inplace(x,y): x *= y
def _div_inplace(x,y): x /= y
def _pow_inplace(x,y): x **= y

def add_MultiThreaded(a, b, threads=NUM_THREADS, direction='left'):
    __MultiThreaded_opr_direction(a, b, _add_inplace, threads, direction=direction)
def sub_MultiThreaded(a, b, threads=NUM_THREADS, direction='left'):
    __MultiThreaded_opr_direction(a, b, _sub_inplace, threads, direction=direction)
def mul_MultiThreaded(a, b, threads=NUM_THREADS, direction='left'):
    __MultiThreaded_opr_direction(a, b, _mul_inplace, threads, direction=direction)
def div_MultiThreaded(a, b, threads=NUM_THREADS, direction='left'):
    __MultiThreaded_opr_direction(a, b, _div_inplace, threads, direction=direction)
def pow_MultiThreaded(a, b, threads=NUM_THREADS, direction='left'):
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

    threads = threads or NUM_THREADS
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
    pass