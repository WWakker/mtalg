import mtalg


def get_num_threads() -> int:
    """Get number of threads for multithreaded RNGs and algebra functions.

    Returns:
        Number of threads.

    Examples:
        Check the available number of threads.

        >>> from multiprocessing import cpu_count
        >>> cpu_count()
        12

        Check the number of threads used by mtalg.

        >>> mtalg.get_num_threads()
        12

        Change the number of threads used by mtalg.

        >>> mtalg.set_num_threads(6)
        >>> mtalg.get_num_threads()
        6
    """
    return mtalg.core.threads._global_num_threads
