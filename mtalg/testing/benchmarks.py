import numpy as np, timeit, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit, prange
from numpy.random import default_rng
sns.set()
from mtalg.alg import (add_MultiThreaded,
                       sub_MultiThreaded,
                       mul_MultiThreaded,
                       div_MultiThreaded,
                       pow_MultiThreaded)
SHAPE = (int(4e4), 1)
def get_a_b(shape=SHAPE):
    rng = default_rng(1)
    a = rng.standard_normal(shape)
    a = a.clip(min=1e-5)
    b = a * 3.14
    return a, b
def get_a(shape=SHAPE):
    rng = default_rng(1)
    a = rng.standard_normal(shape)
    a = a.clip(min=1e-5)
    return a
  
def _add(x,y): return x + y
def _sub(x,y): return x - y
def _mul(x,y): return x * y
def _div(x,y): return x / y
def _pow(x,y): return x ** y

import numexpr as ne
def ne_add(a, b): return ne.evaluate('a + b')
def ne_sub(a, b): return ne.evaluate('a - b')
def ne_mul(a, b): return ne.evaluate('a * b')
def ne_div(a, b): return ne.evaluate('a / b')
def ne_pow(a, b): return ne.evaluate('a ** b')

@njit(parallel=True, fastmath=True)
def numba_add(a, b):
  a = a + b
  return a 
@njit(parallel=True, fastmath=True)
def numba_sub(a, b):
  a = a - b
  return a
@njit(parallel=True, fastmath=True)
def numba_mul(a, b):
  a = a * b
  return a
@njit(parallel=True, fastmath=True)
def numba_div(a, b):
  a = a / b
  return a
@njit(parallel=True, fastmath=True)
def numba_pow(a, b):
  a = a ** b
  return a

#a, b = get_a_b(shape=int(1e10))
#a = get_a(shape=int(1e9))
#import dask.array as da
#x = da.random.random((10_000, 100_000), chunks=(1000, 1000))
#%timeit add_MultiThreaded(a,b)
###################################################
###################################################
###################################################
ADD_FUNCS = {'mtalg': add_MultiThreaded, 'numexpr': ne_add, 
             'numba': numba_add, 'numpy': _add}
  

if __name__=='__main__':
  
  
  TIME = {k:[] for k in ADD_FUNCS.keys()}
  
  LOG_LW, LOG_UP, N = 4.5, 9, 500
  COMPLEXITY = np.logspace(LOG_LW, LOG_UP, N).astype(int)[::-1]
  
  for s in tqdm(COMPLEXITY):
    for lab, func in ADD_FUNCS.items():
      func_name = func.__name__
      ts = timeit.repeat(f"{func_name}(a, b)", 
                         f"from mtalg.testing.benchmarks import {func_name}, get_a_b;a, b = get_a_b(shape={s})",
                          number=1, repeat=3)
      TIME[lab].append(np.min(ts))
      

  
  TIME = pd.DataFrame(TIME, index=COMPLEXITY).sort_index()
  TIME.to_csv('__RES/TIME.csv')
  
  def plot(save=False, path=None):
    TIME.rolling(5).mean().plot(xlabel='Number of operations (size of the array)',
                               ylabel='Execution time [sec]') 
    plt.loglog()
    plt.xlim(TIME.index.min(), TIME.index.max())
    if save:
      plt.tight_layout()
      plt.savefig(f"{path or '__RES'}/benchmark_add.png", dpi=400)
      plt.savefig(f"{path or '__RES'}/benchmark_add.svg")
        
  plot(save=True, path='mtalg/__res/benchmark')
    
  def barplot(save=False, path=None):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    TIME.iloc[-1, :].plot.bar(color=colors, ylabel='Execution time [sec]',
                             title='1bn operations')
    if save:
      plt.tight_layout()
      plt.savefig(f"{path or '__RES'}/benchmark_add_BARS.png", dpi=400)
      plt.savefig(f"{path or '__RES'}/benchmark_add_BARS.svg")
      
  barplot(save=True, path='mtalg/__res/benchmark')



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
