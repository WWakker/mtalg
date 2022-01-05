import mtalg

def get_num_threads():
    """Get number of threads for MRNGs and algebra functions
    Args:
        num_threads: Number of threads
    """
    return mtalg.core.threads._global_num_threads
  
