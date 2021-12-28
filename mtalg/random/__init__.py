from mtalg.random.random_number_generators import MultithreadedRNG
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


beta.__doc__ = _RNG.beta.__doc__
binomial.__doc__ = _RNG.binomial.__doc__
chisquare.__doc__ = _RNG.chisquare.__doc__
exponential.__doc__ = _RNG.exponential.__doc__
f.__doc__ = _RNG.f.__doc__
gamma.__doc__ = _RNG.gamma.__doc__
geometric.__doc__ = _RNG.geometric.__doc__
gumbel.__doc__ = _RNG.gumbel.__doc__
hypergeometric.__doc__ = _RNG.hypergeometric.__doc__
integers.__doc__ = _RNG.integers.__doc__
laplace.__doc__ = _RNG.laplace.__doc__
logistic.__doc__ = _RNG.logistic.__doc__
lognormal.__doc__ = _RNG.lognormal.__doc__
logseries.__doc__ = _RNG.logseries.__doc__
negative_binomial.__doc__ = _RNG.negative_binomial.__doc__
noncentral_chisquare.__doc__ = _RNG.noncentral_chisquare.__doc__
noncentral_f.__doc__ = _RNG.noncentral_f.__doc__
normal.__doc__ = _RNG.normal.__doc__
pareto.__doc__ = _RNG.pareto.__doc__
poisson.__doc__ = _RNG.poisson.__doc__
power.__doc__ = _RNG.power.__doc__
random.__doc__ = _RNG.random.__doc__
rayleigh.__doc__ = _RNG.rayleigh.__doc__
standard_cauchy.__doc__ = _RNG.standard_cauchy.__doc__
standard_exponential.__doc__ = _RNG.standard_exponential.__doc__
standard_gamma.__doc__ = _RNG.standard_gamma.__doc__
standard_normal.__doc__ = _RNG.standard_normal.__doc__
standard_t.__doc__ = _RNG.standard_t.__doc__
triangular.__doc__ = _RNG.triangular.__doc__
uniform.__doc__ = _RNG.uniform.__doc__
vonmises.__doc__ = _RNG.vonmises.__doc__
wald.__doc__ = _RNG.wald.__doc__
weibull.__doc__ = _RNG.weibull.__doc__
zipf.__doc__ = _RNG.zipf.__doc__
