import inspect
from .random_number_generators import MultithreadedRNG

__RNG = MultithreadedRNG()

get_methods = lambda obj: [m for m in dir(obj) if callable(getattr(obj, m)) and m[0]!='_']

for method in get_methods(__RNG):
    argsspec = inspect.getfullargspec(getattr(__RNG, method))
    args_all, defaults = argsspec.args[1:], argsspec.defaults or []
    kwargs = {k:v for k,v in zip(args_all[len(args_all)-len(defaults):], defaults)}
    args = args_all[:len(args_all)-len(defaults)]

    def func(*args, **kwargs):
        getattr(__RNG, method)(*args, **kwargs)
        return __RNG.values

    exec(f'{method}=func')

