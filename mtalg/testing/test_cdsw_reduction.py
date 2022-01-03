import os
from mtalg.random import MultithreadedRNG


def cdsw(func):

    if os.name != 'posix':
        def empty_wrapper(self):
            print("Not on posix")
        return empty_wrapper
    
    def wrapper(self):
        os.environ['CDSW_NODE_NAME'] = 'CDSW_NODE_NAME'
        os.environ['DISC_KRB_REALM'] = 'MIT01.ECB.DE'
        os.environ['CDSW_CPU_MILLICORES'] = '1000'
        func(self)
        for env in ['CDSW_NODE_NAME', 'DISC_KRB_REALM', 'CDSW_CPU_MILLICORES']:
            del os.environ[env]
    
    return wrapper


class TestMRNGCDSW:
    @cdsw
    def test1(self):
        mrng = MultithreadedRNG(seed=1)
        assert mrng._num_threads == 1

    @cdsw
    def test2(self):
        mrng = MultithreadedRNG(seed=1, num_threads=4)
        a = mrng.standard_normal(4)
        assert a.shape == (4,)
        assert mrng._num_threads == 1
