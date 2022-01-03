import pytest, os, numpy as np
from numpy.random import default_rng
from itertools import chain, combinations


def _add(x, y): return x + y
def _sub(x, y): return x - y
def _mul(x, y): return x * y
def _div(x, y): return x / y
def _pow(x, y): return x ** y


def all_unique_combinations(ll):
    return set(chain.from_iterable(combinations(ll, i + 1) for i in range(len(ll))))


SHAPE = (int(4e4), 10, 5)


def get_a_b(size, no_negative=False):
    import mtalg

    rng = default_rng(1)
    aa = rng.standard_normal(size)
    bb = rng.standard_normal(size)
    bb = bb.clip(min=1e-5)
    if no_negative:
        aa = aa.clip(min=1e-5)
    return aa, bb


class TestEmAlgebra:

    def test_missing_numba(self):
        os.system("pip uninstall --yes numba")
        import mtalg
        
        a = np.arange(100)
        with pytest.raises(ModuleNotFoundError):
            mtalg.std(a)
        os.system("pip install numba")
        
    def test_numba_wrapped_functions(self):
        import mtalg
        os.environ['NUMBA_DISABLE_JIT'] = '1'
        a = np.arange(100)
        mtalg.std(a)
        
    def test_all(self):
        import mtalg
        failed = {}
        for op, opMT in zip((_add, _sub, _mul, _div, _pow),
                            (mtalg.add, mtalg.sub, mtalg.mul, mtalg.div, mtalg.pow)):
            for shape in all_unique_combinations(SHAPE):
                for scalar in [False, True]:
                    a, b = get_a_b(shape, no_negative=True) if opMT.__name__ == 'pow' else get_a_b(shape)
                    if scalar:
                        b = 2
                    c = op(a, b)
                    opMT(a, b, direction='left')
                    if not (c[c == c] == a[a == a]).all():
                        failed[(opMT.__name__, shape, scalar)] = 'failed'
                        raise Warning(f'Failed with operation {opMT}; shape arrays: {shape}.')

        assert not failed

    def test1(self):
        import mtalg
        a = np.arange(100)
        b = np.arange(100)
        mtalg.add(a, b, direction='right')

    def test2(self):
        import mtalg
        a = np.arange(100)
        b = np.arange(100)
        with pytest.raises(ValueError):
            mtalg.add(a, b, direction='middle')

    def test3(self):
        import mtalg
        a = np.arange(100)
        assert np.isclose(np.std(a), mtalg.std(a))


    def test4(self):
        import mtalg
        a_arr = np.arange(10000).reshape((100, 100))
        a_scal = 2
        b_arr = np.arange(10000).reshape((100, 100))
        b_scal = 2
        mtalg.add(a_arr, b_arr, direction='left', num_threads=2)
        mtalg.add(a_arr, b_arr, direction='right', num_threads=2)
        mtalg.add(a_arr, b_scal, direction='left', num_threads=2)
        with pytest.raises(ValueError):
            mtalg.add(a_arr, b_scal, direction='right', num_threads=2)
        with pytest.raises(ValueError):
            mtalg.add(a_scal, b_scal, direction='left')
        with pytest.raises(ValueError):
            mtalg.add(a_scal, b_scal, direction='right')
