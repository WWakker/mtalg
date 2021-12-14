import numpy as np
from numpy.random import default_rng
from timeit import timeit
from mtalg.random import MultithreadedRNG
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import mkl_random
sns.set()

mrng = MultithreadedRNG(seed=1)
rng = default_rng(seed=1)
result = {'x': [], 'MT_std_norm':[], 'np_std_norm': [],
         'MT_uniform': [], 'np_uniform': []}
for x in tqdm(np.geomspace(1, 1e9, num=300)):
    result['x'].append(x) 
    result['MT_std_norm'].append(timeit(lambda: mrng.standard_normal(size=int(x)), number=10) / 10)
    result['np_std_norm'].append(timeit(lambda: rng.standard_normal(size=int(x)), number=10) / 10)
    result['MT_uniform'].append(timeit(lambda: mrng.uniform(size=int(x)), number=10) / 10)
    result['np_uniform'].append(timeit(lambda: rng.uniform(size=int(x)), number=10) / 10)

df = pd.DataFrame(result).set_index('x')
#df.to_csv('__RES/df_rng.csv')
df_plot = df.rolling(40).mean()

def plot_line(save=False, path=None):
    fig, ax = plt.subplots()
    for key, label, color, lstyle in zip(['MT_std_norm', 'np_std_norm', 'MT_uniform', 'np_uniform'], 
                          ['mtalg - std normal', 'Numpy - std normal', 'mtalg - uniform', 'Numpy - uniform'],
                          ['b', 'r', 'b', 'r'], ['-', '-', '--', '--']):
        ax.plot(df_plot.index, df_plot[key], label=label, color=color, linestyle=lstyle)
    ax.legend()
    ax.set_xlabel('Number of operations (size of the array)')
    ax.set_ylabel('Execution time [sec]')
    plt.loglog()
    plt.xlim(1e3, df_plot.index.max())
    
    if save:
        plt.tight_layout()
        plt.savefig(f"{path or 'mtalg/__res/benchmark'}/benchmark_rng.png", dpi=400)
        plt.savefig(f"{path or 'mtalg/__res/benchmark'}/benchmark_rng.svg")
    
plot_line(save=True)


def plot_bar(save=False, path=None):
    x = np.array([0, 1])
    width = .35
    
    fig, ax = plt.subplots()
    ax.bar(x - width/2, [df_plot['MT_std_norm'].values[-1], df_plot['MT_uniform'].values[-1]], color='b', width=width, label='mtalg')
    ax.bar(x + width/2, [df_plot['np_std_norm'].values[-1], df_plot['np_uniform'].values[-1]], color='r', width=width, label='Numpy')
    ax.set_xticks(x)
    ax.set_xticklabels(['Standard normal', 'Uniform'])
    ax.set_ylabel('Execution time [sec]')
    ax.set_title('1bn operations')
    ax.legend()
    
    if save:
        plt.tight_layout()
        plt.savefig(f"{path or 'mtalg/__res/benchmark'}/benchmark_rng_BAR.png", dpi=400)
        plt.savefig(f"{path or 'mtalg/__res/benchmark'}/benchmark_rng_BAR.svg")
    
plot_bar(save=True)
