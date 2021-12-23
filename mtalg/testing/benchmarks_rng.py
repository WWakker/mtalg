import numpy as np
from numpy.random import default_rng
from timeit import timeit
from mtalg.random import MultithreadedRNG
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import mkl_random
from collections import defaultdict
sns.set()

mrng = MultithreadedRNG(seed=1, num_threads=24)
rng = default_rng(seed=1)
result = defaultdict(list)

if __name__ == '__main__':

    for x in tqdm(np.geomspace(1e3, 1e9, num=200)[::-1]):
        n = 20
        result['x'].append(x)
        result['MT_std_norm'].append(timeit(lambda: mrng.standard_normal(size=int(x)), number=n) / n)
        result['np_std_norm'].append(timeit(lambda: rng.standard_normal(size=int(x)), number=n) / n)
        result['mkl_std_norm'].append(timeit(lambda: mkl_random.standard_normal(size=int(x)), number=n) / n)
        result['MT_uniform'].append(timeit(lambda: mrng.uniform(size=int(x)), number=n) / n)
        result['np_uniform'].append(timeit(lambda: rng.uniform(size=int(x)), number=n) / n)
        result['mkl_uniform'].append(timeit(lambda: mkl_random.uniform(size=int(x)), number=n) / n)

    df = pd.DataFrame(result).set_index('x').sort_index()
    df.to_parquet('result.parq')
    df_plot = df.rolling(10).mean()

    def plot_line(save=False, path=None):
        fig, ax = plt.subplots()
        for key, label, color, lstyle, lw in zip(
                ['MT_std_norm', 'mkl_std_norm', 'np_std_norm', 'MT_uniform', 'mkl_uniform', 'np_uniform'],
                ['mtalg - standard normal', 'mkl_random - standard_normal', 'numpy - standard normal',
                 'mtalg - uniform', 'mkl_random - uniform', 'numpy - uniform'],
                ['b', 'r', 'Y', 'b', 'r', 'Y'],
                ['-', '-', '-', '--', '--', '--'],
                [3, 1.2, 1.2, 3, 1.2, 1.2]):
            ax.plot(df_plot.index, df_plot[key], label=label, color=color, linestyle=lstyle, linewidth=lw)
        ax.legend(loc='upper left', frameon=False)
        ax.set_xlabel('Number of operations (size of the array)')
        ax.set_ylabel('Execution time [sec]')
        plt.loglog()
        plt.xlim(1e5, df_plot.index.max())
        plt.ylim(1e-4, 1e2)

        if save:
            plt.tight_layout()
            fig.savefig(f"{path or 'mtalg/__res/benchmark'}/benchmark_rng.png", dpi=400)
            fig.savefig(f"{path or 'mtalg/__res/benchmark'}/benchmark_rng.svg")

    plot_line(save=False)


    def plot_bar(save=False, path=None):
        x = np.array([0, 1])
        width = .25

        fig, ax = plt.subplots()
        ax.bar(x - width, [df_plot[x].values[-1] for x in ['MT_std_norm', 'MT_uniform']], color='b', width=width,
               label='mtalg')
        ax.bar(x, [df_plot[x].values[-1] for x in ['mkl_std_norm', 'mkl_uniform']], color='r', width=width,
               label='mkl_random')
        ax.bar(x + width, [df_plot[x].values[-1] for x in ['np_std_norm', 'np_uniform']], color='Y', width=width,
               label='numpy')
        ax.set_xticks(x)
        ax.set_xticklabels(['Standard normal', 'Uniform'])
        ax.set_ylabel('Execution time [sec]')
        ax.set_title('1bn operations')
        ax.legend()

        if save:
            plt.tight_layout()
            fig.savefig(f"{path or 'mtalg/__res/benchmark'}/benchmark_rng_BAR.png", dpi=400)
            fig.savefig(f"{path or 'mtalg/__res/benchmark'}/benchmark_rng_BAR.svg")

    plot_bar(save=True)
