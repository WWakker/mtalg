import mtalg


def get_num_threads() -> int:
    """Get number of threads for multithreaded RNGs and algebra functions

    Args
        num_threads: Number of threads

    Returns
        Number of threads
    """
    return mtalg.core.threads._global_num_threads
