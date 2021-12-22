import matplotlib.pyplot as plt
import numexpr as ne
import numpy as np
import pandas as pd
import seaborn as sns
import timeit
from numba import njit, prange
from tqdm import tqdm
from mtalg import add

sns.set()

rng = np.random.default_rng()


def _add(x, y): return x + y


def _sub(x, y): return x - y


def _mul(x, y): return x * y


def _div(x, y): return x / y


def _pow(x, y): return x ** y


def ne_add(a, b): return ne.evaluate('a + b')


def ne_sub(a, b): return ne.evaluate('a - b')


def ne_mul(a, b): return ne.evaluate('a * b')


def ne_div(a, b): return ne.evaluate('a / b')


def ne_pow(a, b): return ne.evaluate('a ** b')


@njit(parallel=True)
def numba_add(a, b):
    for i in prange(len(a)):
        a[i] += b[i]


@njit(parallel=True)
def numba_sub(a, b):
    for i in prange(len(a)):
        a[i] -= b[i]


@njit(parallel=True)
def numba_mul(a, b):
    for i in prange(len(a)):
        a[i] *= b[i]


@njit(parallel=True)
def numba_div(a, b):
    for i in prange(len(a)):
        a[i] /= b[i]


@njit(parallel=True)
def numba_pow(a, b):
    for i in prange(len(a)):
        a[i] **= b[i]


if __name__ == '__main__':

    ########
    # Charts

    def get_a_b(shape):
        a = rng.standard_normal(shape)
        a = a.clip(min=1e-5)
        b = a * 3.14
        return a, b


    result = {'x': [], 'mtalg': [], 'numexpr': [],
              'numba': [], 'numpy': []}

    for x in tqdm(np.geomspace(1e3, 1e9, num=200)[::-1].astype(int)):
        a, b = get_a_b(shape=x)
        n = 50
        result['x'].append(x)
        result['mtalg'].append(timeit.timeit(lambda: add(a, b, num_threads=24), number=n) / n)
        result['numexpr'].append(timeit.timeit(lambda: ne_add(a, b), number=n) / n)
        result['numba'].append(timeit.timeit(lambda: numba_add(a, b), number=n) / n)
        result['numpy'].append(timeit.timeit(lambda: _add(a, b), number=n) / n)

    df = pd.DataFrame(result).set_index('x').sort_index()
    df_plot = df.rolling(10).mean()


    def plot_line(save=False, path=None):
        fig, ax = plt.subplots()
        for key, color, lw in zip(['mtalg', 'numba', 'numexpr', 'numpy'],
                                  ['b', 'r', 'g', 'Y'],
                                  [3, 1.2, 1.2, 1.2]):
            ax.plot(df_plot.index,
                    df_plot[key],
                    label=key,
                    color=color,
                    linewidth=lw)
        ax.legend()
        ax.set_xlabel('Number of operations (size of the array)')
        ax.set_ylabel('Execution time [sec]')
        plt.loglog()
        plt.xlim(1e5, df_plot.index.max())

        if save:
            plt.tight_layout()
            fig.savefig(f"{path or 'mtalg/__res/benchmark'}/benchmark_add.png", dpi=400)
            fig.savefig(f"{path or 'mtalg/__res/benchmark'}/benchmark_add.svg")


    plot_line(save=False)


    def plot_bar(save=False, path=None):
        width = .5
        bars = ['mtalg', 'numba', 'numexpr', 'numpy']

        fig, ax = plt.subplots()
        ax.bar([0, 1, 2, 3],
               [df_plot[bar].values[-1] for bar in bars],
               color=['b', 'r', 'g', 'y'],
               width=width)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(bars)
        ax.set_ylabel('Execution time [sec]')
        ax.set_title('1bn operations')

        if save:
            plt.tight_layout()
            fig.savefig(f"{path or 'mtalg/__res/benchmark'}/benchmark_add_BARS.png", dpi=400)
            fig.savefig(f"{path or 'mtalg/__res/benchmark'}/benchmark_add_BARS.svg")


    plot_bar(save=False)
