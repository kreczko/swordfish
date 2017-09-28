"""`swordfish` is Python tool to study the information yield of counting experiments.

Motivation
----------

The main purpose of `swordfish` is make the world a better place.  It does so
by quickly and accurately forecasting experimental sensitivites without all the
fuss with time-intensive Monte Carlos, mock data generation and likelihood
maximization.

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


Installation
------------

The following dependencies are required:

- whatever


`swordfish` can be installed by invocing

    git clone
    cd swordfish
    python setup.py install


Citation
--------

    TODO
"""

from swordfish.core import *
from swordfish import metricplot
__all__ = ["Swordfish", "EffectiveCounts", "Funkfish", "metricplot"]
