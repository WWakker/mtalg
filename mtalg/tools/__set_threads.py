import mtalg


def set_num_threads(num_threads: int):
    """Set number of threads for subsequent MRNGs and algebra functions

    Args:
        num_threads: Number of threads
    """
    if not isinstance(num_threads, int):
        raise ValueError(f'Number of threads must be an integer, found: {num_threads}')
    if not num_threads > 0:
        raise ValueError(f'Number of threads must be > 0, found: {num_threads}')

    mtalg.core.threads._global_num_threads = num_threads
    mtalg.random._RNG = mtalg.random.MultithreadedRNG(num_threads=num_threads)
