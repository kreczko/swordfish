#!/usr/bin/env python

"""
File: DSspectra.py
Author: Christoph Weniger
Email: c.weniger@uva.nl
Description: Simple wrapper for DS spectra
Date: 2014
"""

from __future__ import division
from numpy import *
import pylab as plt
from ctypes import *

def _initialize():
    global _pHandle
    # Load DarkSUSY shared library
    try:
        _pHandle = cdll.LoadLibrary('libdarksusy.so')
    except OSError as e:
        print "Shared DS version not found."
        print e
        return False

    # Initialize DarkSUSY
    _pHandle.dsinit_()

    # Set return type
    _pHandle.dshayield_.restype = c_double

_initialize()

class spec(object):
    """docstring for spec"""
    def __init__(self, channel=None, mass=None, type=None):
        self.channel = channel
        self.type = type
        self.mass = c_double(mass)
        self.yieldk = c_int()
        self.ch = c_int()
        self.istat = c_int(0)
        self.channelMap = {
            'ZZ':    12,
            'WW':    13,
            'ee':    15,
            'mm':    17,
            'tautau': 19,
            'uu':    20,
            'dd':    21,
            'cc':    22,
            'ss':    23,
            'tt':    24,
            'bb':    25,
            'gg':    26
            }
        self.typeMap = {
            'gam': 152,
            'pos': 151,
            'pbar': 154
            }
        self._setNumbers()

    def _setNumbers(self):
        if self.channel in self.channelMap.keys():
            self.ch = c_int(self.channelMap[self.channel])
        else:
            raise KeyError("Channel unknown.")
        if self.type in self.typeMap.keys():
            self.yieldk = c_int(self.typeMap[self.type])
        else:
            raise KeyError("Type unknown.")

    def _value(self, energy):
        energy = c_double(energy)
        return _pHandle.dshayield_(byref(self.mass), byref(energy), byref(self.ch), byref(self.yieldk), byref(self.istat))

    def __call__(self, energy):
        return vectorize(lambda x: self._value(x))(energy)


# INFOS FROM DARKSUSY DSHAYIELD.F

    # Type
    # *** particle       integrated yield     differential yield
    # *** --------       ----------------     ------------------
    # *** positron                     51                    151
    # *** cont. gamma                  52                    152
    # *** nu_mu and nu_mu-bar          53                    153
    # *** antiproton                   54                    154
    # *** cont. gamma w/o pi0          55                    155
    # *** nu_e and nu_e-bar            56                    156
    # *** nu_tau and nu_tau-bar        57                    157
    # *** pi0                          58                    158
    # *** nu_mu and nu_mu-bar          71                    171 (same as 53/153)
    # *** muons from nu at creation    72                    172
    # *** muons from nu at detector    73                    173

    # Channel
    # *** Ch No  Particles                 Old Ch No   Old chi   New chcomp
    # *** -----  ---------                 ---------   -------   ----------
    # ***  1     S1 S1                      -          -         Not done yet
    # ***  2     S1 S2                      -          -
    # ***  3     S2 S2                      -          -
    # ***  4     S3 S3                      -          -
    # ***  5     S1 S3                      7          -
    # ***  6     S2 S3                      11         -
    # ***  7     S- S+                      -          -
    # ***  8     S1 Z                       8          -
    # ***  9     S2 Z                       9          -
    # *** 10     S3 Z	                      -          -
    # *** 11     W- S+ and W+ S-            10         -
    # *** 12     Z0 Z0 	              6          6
    # *** 13     W+ W-                      5          5
    # *** 14     nu_e nu_e-bar              -          -
    # *** 15     e+ e-                      -          -
    # *** 16     nu_mu nu_mu-bar            -          -
    # *** 17     mu+ mu-                    13         7
    # *** 18     nu_tau nu_tau-bar	      -          -
    # *** 19     tau+ tau-	              4          4
    # *** 20     u u-bar                    -          -
    # *** 21     d d-bar                    -          -
    # *** 22     c c-bar                    1          1
    # *** 23     s s-bar                    -          -
    # *** 24     t t-bar                    3          3
    # *** 25     b b-bar                    2          2
    # *** 26     gluon gluon                12         8
    # *** 27     q q gluon (not implemented yet, put to zero)
    # *** 28     gamma gamma (1-loop)
    # *** 29     Z0 gamma (1-loop)


def unitTest():
    for c in ['bb', 'ss', 'tt', 'ss', 'uu', 'dd', 'tautau', 'ZZ', 'mm', 'ee']:
        s = spec(mass = 400, type='gam', channel=c)
        x = logspace(0, 3, 1000)
        y = s(x)
        plt.loglog(x, y*x*x)
    plt.show()

if __name__ == '__main__':
    unitTest()
