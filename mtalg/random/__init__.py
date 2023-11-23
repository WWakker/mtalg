from mtalg.random.random_number_generators import MultithreadedRNG
from inspect import signature
_RNG = MultithreadedRNG()


def beta(*args, **kwargs):
    return _RNG.beta(*args, **kwargs)


def binomial(*args, **kwargs):
    return _RNG.binomial(*args, **kwargs)


def chisquare(*args, **kwargs):
    return _RNG.chisquare(*args, **kwargs)


def exponential(*args, **kwargs):
    return _RNG.exponential(*args, **kwargs)


def f(*args, **kwargs):
    return _RNG.f(*args, **kwargs)


def gamma(*args, **kwargs):
    return _RNG.gamma(*args, **kwargs)


def geometric(*args, **kwargs):
    return _RNG.geometric(*args, **kwargs)


def gumbel(*args, **kwargs):
    return _RNG.gumbel(*args, **kwargs)


def hypergeometric(*args, **kwargs):
    return _RNG.hypergeometric(*args, **kwargs)


def integers(*args, **kwargs):
    return _RNG.integers(*args, **kwargs)


def laplace(*args, **kwargs):
    return _RNG.laplace(*args, **kwargs)


def logistic(*args, **kwargs):
    return _RNG.logistic(*args, **kwargs)


def lognormal(*args, **kwargs):
    return _RNG.lognormal(*args, **kwargs)


def logseries(*args, **kwargs):
    return _RNG.logseries(*args, **kwargs)


def negative_binomial(*args, **kwargs):
    return _RNG.negative_binomial(*args, **kwargs)


def noncentral_chisquare(*args, **kwargs):
    return _RNG.noncentral_chisquare(*args, **kwargs)


def noncentral_f(*args, **kwargs):
    return _RNG.noncentral_f(*args, **kwargs)


def normal(*args, **kwargs):
    return _RNG.normal(*args, **kwargs)


def pareto(*args, **kwargs):
    return _RNG.pareto(*args, **kwargs)


def poisson(*args, **kwargs):
    return _RNG.poisson(*args, **kwargs)


def power(*args, **kwargs):
    return _RNG.power(*args, **kwargs)


def random(*args, **kwargs):
    return _RNG.random(*args, **kwargs)


def rayleigh(*args, **kwargs):
    return _RNG.rayleigh(*args, **kwargs)


def standard_cauchy(*args, **kwargs):
    return _RNG.standard_cauchy(*args, **kwargs)


def standard_exponential(*args, **kwargs):
    return _RNG.standard_exponential(*args, **kwargs)


def standard_gamma(*args, **kwargs):
    return _RNG.standard_gamma(*args, **kwargs)


def standard_normal(*args, **kwargs):
    return _RNG.standard_normal(*args, **kwargs)


def standard_t(*args, **kwargs):
    return _RNG.standard_t(*args, **kwargs)


def triangular(*args, **kwargs):
    return _RNG.triangular(*args, **kwargs)


def uniform(*args, **kwargs):
    return _RNG.uniform(*args, **kwargs)


def vonmises(*args, **kwargs):
    return _RNG.vonmises(*args, **kwargs)


def wald(*args, **kwargs):
    return _RNG.wald(*args, **kwargs)


def weibull(*args, **kwargs):
    return _RNG.weibull(*args, **kwargs)


def zipf(*args, **kwargs):
    return _RNG.zipf(*args, **kwargs)


for func in [beta, binomial, chisquare, exponential, f, gamma, geometric, gumbel, hypergeometric, integers, laplace,
             logistic, lognormal, logseries, negative_binomial, noncentral_chisquare, noncentral_f, normal, pareto,
             poisson, power, random, rayleigh, standard_cauchy, standard_exponential, standard_gamma, standard_normal,
             standard_t, triangular, uniform, vonmises, wald, weibull, zipf]:
    func.__doc__ = getattr(_RNG, func.__name__).__doc__
    func.__signature__ = signature(getattr(_RNG, func.__name__))
