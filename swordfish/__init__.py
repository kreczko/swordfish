"""`swordfish` is Python tool to study the information yield of counting experiments.

A large range of experiments in particle physics and astronomy are at their
likelihood level described by a Poisson point process.  The `swordfish` module
implements a general class of such Poisson point processes, and provides easy
access to its information geometrical properties.  This allows to efficiently
study and optimize the expected information gain of an experiment from many
different perspectives.

In particular, with `swordfish` you can do the following.

- Signal detection, using *effective counts method*
    - Expected upper limits
    - Expected discovery threshold
- Parameter regression, using *information geometry*
    - Confidence contours (2-D)
    - Streamline visualization of Fisher metric (2-D)
- Misc
    - Handle general covariance matrix for background systematics in event space
    - Integration with `iminuit` to double-check accuracy.
- Experimental design, using *information flux*
    - Signal-to-noise maps in event space
    - Effective signal-to-noise maps that account for correlations and
      background systematics


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
