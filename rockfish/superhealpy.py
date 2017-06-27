#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import healpy as hp
import pylab as pl
from copy import copy, deepcopy
from scipy import sparse as sp

class HealpixMap():
    def __init__(self, nside = None, data = None, ipix = None, verbose = True):
        self.nside = nside
        self.data = data if data is not None else np.empty((0,), dtype=np.float64)
        self.ipix = ipix if ipix is not None else np.empty((0,), dtype=np.int64)
        self.verbose = verbose

    def add_disc(self, vec, radius):
        if self.verbose: print "add_disc"
        ipix = hp._query_disc.query_disc(self.nside, vec, radius, nest=True)
        self.ipix = np.append(self.ipix, ipix)
        self.data = np.append(self.data, np.zeros(len(ipix)))
        self.clean()

    def add_polygon(self, vertices):
        if self.verbose: print "add_polygon"
        ipix = hp._query_disc.query_polygon(self.nside, vertices, nest=True)
        self.ipix = np.append(self.ipix, ipix)
        self.data = np.append(self.data, np.zeros(len(ipix)))
        self.clean()

    def full(self):
        if self.verbose: print "full"
        npix = hp.nside2npix(self.nside)
        m = np.zeros(npix)
        # Note: np.add.at treats repeated indices correctly
        np.add.at(m, self.ipix, self.data)  
        return m

    def clean(self):
        if self.verbose: print "clean"
        ipix, inverse = np.unique(self.ipix, return_inverse = True)
        data = np.zeros(len(ipix))
        np.add.at(data, inverse, self.data)  
        self.ipix = ipix
        self.data = data

    def rebinned(self, nside = None):
        if self.verbose: print "rebinned"
        if nside < self.nside:
            shift = 2*int(np.log2(self.nside/nside))
            ratio = (self.nside/nside)**2
            ipix = self.ipix >> shift
            data = self.data/ratio
            nside = nside
            return HealpixMap(nside, data, ipix)
        elif nside > self.nside:
            shift = 2*int(np.log2(nside/self.nside))
            factor = int((nside/self.nside)**2)
            npix_orig = len(self.ipix)
            ipix = np.repeat(self.ipix << shift, factor)
            ipix += np.tile(np.arange(factor), npix_orig)
            data = np.repeat(self.data, factor)
            nside = nside
            return HealpixMap(nside, data, ipix)
        else:
            return deepcopy(self)

    def __imul__(self, other):
        if self.nside == other.nside:
            o = other
        else:
            o = other.rebinned(nside = self.nside)
        data_con = np.concatenate((self.data, o.data))
        ipix = np.concatenate((self.ipix, o.ipix))
        ipix, inverse, counts = np.unique(ipix, return_inverse = True,
                return_counts = True)
        mask = counts == 2
        data = np.ones_like(ipix)
        np.multiply.at(data, inverse, data_con)
        self.data = data[mask]
        self.ipix = ipix[mask]
        return self

    def __iadd__(self, other):
        if self.nside == other.nside:
            o = other
        else:
            o = other.rebinned(nside = self.nside)
        self.data = np.concatenate((self.data, o.data))
        self.ipix = np.concatenate((self.ipix, o.ipix))
        self.clean()
        return self

    def __mul__(self, other):
        """Add to dense map."""
        if self.verbose: print "mul"
        if isinstance(other, HealpixMap):
            if other.nside >= self.nside:
                h = deepcopy(self)
                h *= other
                return h
            else:
                h = deepcopy(other)
                h *= self
                return h
        else:
            raise NotImplementedError

    def __add__(self, other):
        """Add to dense map."""
        if self.verbose: print "add"
        if isinstance(other, HealpixMap):
            if other.nside >= self.nside:
                h = deepcopy(self)
                h += other
                return h
            else:
                h = deepcopy(other)
                h += self
                return h
        else:
            raise NotImplementedError

h1 = HealpixMap(nside = 1024)
h1.add_disc((1, 0, 0), 0.2)
h1.data += 1

h2 = HealpixMap(nside = 64)
h2.add_polygon(((1, 0, 0), (1, 1, 0), (1, 0, 1)))
h2.add_polygon(((1, 0, 0), (1, -1, 0), (1, 0, 1)))
h2.data += 1
h2.add_disc((1, -1, 0), 0.2)
h2.data += 1

h = h2 * h1 + h2

m = h.full()
hp.mollview(m, nest = True)
pl.savefig('test.eps')
