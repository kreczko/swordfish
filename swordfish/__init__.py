"""`swordfish` is a Python tool to study the information yield of counting experiments.

NOTE: The package is stable, but still in beta phase.  Use for production only
if you know what you are doing.

Motivation
----------

With `swordfish` you can quickly and accurately forecast experimental
sensitivities without all the fuss with time-intensive Monte Carlos, mock data
generation and likelihood maximization.

With `swordfish` you can

- Calculate the expected upper limit or discovery reach of an instrument.
- Derive expected confidence contours for parameter reconstruction.
- Visualize confidence contours as well as the underlying information metric field.
- Calculate the *information flux*, an effective signal-to-noise ratio that
  accounts for background systematics and component degeneracies.

A large range of experiments in particle physics and astronomy are
statistically described by a Poisson point process.  The `swordfish` module
implements at its core a rather general version of a Poisson point process, and
provides easy access to its information geometrical properties.  Based on this
information, a number of common and less common tasks can be performed.


Documentation
-------------

Documentation of `swordfish` can be found on
[github.io](https://cweniger.github.io/swordfish).  For extensive details about
Fisher forecasting with Poisson likelihoods, the effective counts method, the
definition of information flux and the treatment of background systematics see
[http://arxiv.org/abs/1704.05458](http://arxiv.org/abs/1704.05458).


Installation
------------

`swordfish` has been tested with Python 2.7.13 and the packages

- `numpy 1.13.1`
- `scipy 0.19.0`
- `matplotlib 2.0.0`

Let us know if you run into problems.

`swordfish` can be installed by invocing

    git clone https://github.com/cweniger/swordfish
    cd swordfish
    python setup.py install


Citation
--------

Please cite
[http://arxiv.org/abs/1704.05458](http://arxiv.org/abs/1704.05458).

"""

from swordfish.core import *
from swordfish import metricplot
__all__ = ["Swordfish", "EffectiveCounts", "Funkfish", "metricplot"]
