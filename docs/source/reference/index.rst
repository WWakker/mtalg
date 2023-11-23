=============
API reference
=============

.. currentmodule:: mtalg

Multithreading settings
~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   get_num_threads
   set_num_threads

Linear algebra
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   add
   div
   mul
   pow
   sub

Pseudorandom number generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Constructor
-----------
.. autosummary::
   :toctree: api/
   :template: autosummary/class.rst

   random.MultithreadedRNG

Functions
-----------

The functions below are the same as the methods of the :doc:`MultithreadedRNG <api/mtalg.random.MultithreadedRNG>`
defined above, as it comes from an instance of this class, instantiated with default arguments and seed as set by
:std:doc:`numpy:reference/random/generated/numpy.random.seed`.

.. note::
   For more information on distributions, see
   `Random generator distributions <https://numpy.org/doc/stable/reference/random/generator.html#distributions>`_.

.. autosummary::
   :toctree: api/

   random.beta
   random.binomial
   random.chisquare
   random.exponential
   random.f
   random.gamma
   random.geometric
   random.gumbel
   random.hypergeometric
   random.integers
   random.laplace
   random.logistic
   random.lognormal
   random.logseries
   random.negative_binomial
   random.noncentral_chisquare
   random.noncentral_f
   random.normal
   random.pareto
   random.poisson
   random.power
   random.random
   random.rayleigh
   random.standard_cauchy
   random.standard_exponential
   random.standard_gamma
   random.standard_normal
   random.standard_t
   random.triangular
   random.uniform
   random.vonmises
   random.wald
   random.weibull
   random.zipf
