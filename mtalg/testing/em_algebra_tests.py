import mtalg
import numpy as np
from numpy.random import default_rng
from itertools import permutations, chain, combinations


def _add(x, y): return x + y
def _sub(x, y): return x - y
def _mul(x, y): return x * y
def _div(x, y): return x / y
def _pow(x, y): return x ** y


def all_unique_combinations(L):
    return set(chain.from_iterable(combinations(L, i + 1) for i in range(len(L))))


SHAPE = (int(4e4), 10, 5)


def get_a_b(shape=SHAPE):
    rng = default_rng(1)
    a = rng.standard_normal(shape)
    b = rng.standard_normal(shape)
    b = b.clip(min=1e-5)
    return a, b


a, b = get_a_b((2, 2))
c = a ** b
mtalg.pow(a, b, direction='left')
(c[c == c] == a[a == a]).all()

FAILED = {}
for op, opMT in zip((_add, _sub, _mul, _div, _pow),
                    (mtalg.add, mtalg.sub, mtalg.mul, mtalg.div, mtalg.pow)):
    for shape in all_unique_combinations(SHAPE):
        for scalar in [False, True]:
            a, b = get_a_b()
            if scalar:
                b = 2
            c = op(a, b)
            opMT(a, b, direction='left')
            if not (c[c == c] == a[a == a]).all():
                FAILED[(opMT.__name__, shape, scalar)] = 'FAILED'
                raise Warning(f'Failed with operation {opMT}; shape arrays: {shape}.')

if FAILED:
    raise RuntimeError(f"Testing of element-wise algebra failed:\n {FAILED}")
else:
    print("All element-wise algebra tests successfully passed.")
