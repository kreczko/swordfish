#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import healpy as hp
import pylab as plt
import scipy.sparse as sp
from copy import deepcopy

class ARPix():
    def __init__(self, verbose = True):
        self.ipix = np.empty((0,), dtype=np.int64)
        self.order = np.empty((0,), dtype=np.int8)
        self.data = np.empty((0,), dtype=np.float64)
        self.verbose = verbose

    def add_ipix(self, ipix, order, clean = True, fill = 0., insert = False):
        if insert:
            self.ipix = np.append(ipix, self.ipix)
            self.order = np.append(order, self.order)
            self.data = np.append(np.ones(len(ipix))*fill, self.data)
        else:
            self.ipix = np.append(self.ipix, ipix)
            self.order = np.append(self.order, order)
            self.data = np.append(self.data, np.ones(len(ipix))*fill)
        if clean:
            self.clean()

    def add_disc(self, vec, radius, nside, clean = True, fill = 0.):
        order = hp.nside2order(nside)
        ipix = hp._query_disc.query_disc(nside, vec, radius, nest=True)
        self.ipix = np.append(self.ipix, ipix)
        self.data = np.append(self.data, np.ones(len(ipix))*fill)
        self.order = np.append(self.order, order*np.ones(len(ipix), dtype=np.int8))
        if self.verbose: print "add_disc:", len(ipix)
        if clean:
            self.clean()

    def add_polygon(self, vertices, nside, clean = True, fill = 0.):
        order = hp.nside2order(nside)
        ipix = hp._query_disc.query_polygon(nside, vertices, nest=True)
        self.ipix = np.append(self.ipix, ipix)
        self.data = np.append(self.data, np.ones(len(ipix))*fill)
        self.order = np.append(self.order, order*np.ones(len(ipix), dtype=np.int8))
        if self.verbose: print "add_polygon:", len(ipix)
        if clean:
            self.clean()

    def map_to_healpix(self, nside):
        if self.verbose: print "full"
        npix = hp.nside2npix(nside)
        fullorder = hp.nside2order(nside)
        fullmap = np.zeros(npix)
        N = len(self.data)

        # COO matrix setup
        row =  []
        col =  []
        data = []
        shape = (npix, N)
        num = np.arange(N)

        for o in np.unique(self.order):
            mask = self.order == o
            if o > fullorder:
                idx = self.ipix[mask] >> (o-fullorder)*2
                dat = np.ones(len(idx)) / 4**(o-fullorder)
                row.extend(idx)
                col.extend(num[mask])
                data.extend(dat)
            elif o == fullorder:
                idx = self.ipix[mask]
                dat = np.ones(len(idx))
                row.extend(idx)
                col.extend(num[mask])
                data.extend(dat)
            elif o < fullorder:
                idx = self.ipix[mask] << -(o-fullorder)*2
                dat = np.ones(len(idx))
                for i in range(0, 4**(fullorder-o)):
                    row.extend(idx+i)
                    col.extend(num[mask])
                    data.extend(dat)

        M = sp.coo_matrix((data, (row, col)), shape = shape)
        M = M.tocsr()
        return M

    def to_heaplpix(self, nside):
        M = self.map_to_healpix(nside)
        return M.dot(self.data)

    def clean(self):
        if self.verbose: print "clean"
        orders = np.unique(self.order)
        clean_ipix = []
        clean_data = []
        clean_order = []

        for o in np.arange(min(orders), max(orders)+1):
            mask = self.order == o
            maskS = self.order > o

            sub_ipix = self.ipix[maskS] >> 2*(self.order[maskS] - o)

            if o > orders[0]:
                unsubbed = np.in1d(spill_ipix, sub_ipix, invert = True)
                clean_ipix1 = spill_ipix[unsubbed]
                clean_data1 = spill_data[unsubbed]
                spill_ipix1 = np.repeat(spill_ipix[~unsubbed] << 2, 4)
                spill_ipix1 += np.tile(np.arange(4), int(len(spill_ipix1)/4))
                spill_data1 = np.repeat(spill_data[~unsubbed], 4)
            else:
                clean_ipix1 = np.empty((0,), dtype=np.int64)
                clean_data1 = np.empty((0,), dtype=np.float64)
                spill_ipix1 = np.empty((0,), dtype=np.int64)
                spill_data1 = np.empty((0,), dtype=np.float64)

            unsubbed = np.in1d(self.ipix[mask], sub_ipix, invert = True)
            clean_ipix2 = self.ipix[mask][unsubbed]
            clean_data2 = self.data[mask][unsubbed]
            spill_ipix2 = np.repeat(self.ipix[mask][~unsubbed] << 2, 4)
            spill_ipix2 += np.tile(np.arange(4), int(len(spill_ipix2)/4))
            spill_data2 = np.repeat(self.data[mask][~unsubbed], 4)

            clean_ipix_mult = np.append(clean_ipix1, clean_ipix2)
            clean_data_mult = np.append(clean_data1, clean_data2)
            clean_ipix_sing, inverse = np.unique(clean_ipix_mult,
                    return_inverse = True)
            clean_data_sing = np.zeros(len(clean_ipix_sing))
            np.add.at(clean_data_sing, inverse, clean_data_mult)
            clean_ipix.extend(clean_ipix_sing)
            clean_data.extend(clean_data_sing)
            clean_order.extend(np.ones(len(clean_ipix_sing), dtype=np.int8)*o)

            spill_ipix = np.append(spill_ipix1, spill_ipix2)
            spill_data = np.append(spill_data1, spill_data2)

        self.ipix = np.array(clean_ipix)
        self.data = np.array(clean_data)
        self.order = np.array(clean_order)

    def __iadd__(self, other):
        print "+:", len(self.data), len(other.data)
        self.data = np.append(self.data, other.data)
        self.ipix = np.append(self.ipix, other.ipix)
        self.order = np.append(self.order, other.order)
        self.clean()
        print "+:", len(self.data)
        return self

    def __mul__(self, other):
        if isinstance(other, ARPix):
            h1 = deepcopy(self)
            h2 = deepcopy(other)
            h1.add_ipix(other.ipix, other.order, insert=True)
            h2.add_ipix(self.ipix, self.order)
            h1.data *= h2.data
            return h1
        else:
            raise NotImplementedError

    def __add__(self, other):
        """Add to dense map."""
        if self.verbose: print "add"
        if isinstance(other, ARPix):
            h = deepcopy(self)
            h += other
            return h
        else:
            raise NotImplementedError

def test():
    h = ARPix()
    h.add_disc((1, -0.1, 0), 10.4, 1, fill = 1)
    for i in range(100):
        print i
        h1 = ARPix()
        vec = np.random.random(3)*2-1
        h1.add_disc(vec, 0.01, 1024, fill = 1)
        h1.add_disc(vec, 0.1, 16, fill = 1)
        h += h1
    h.data = np.random.random(len(h.data))
    m = h.to_heaplpix(256)
    hp.mollview(m, nest = True, cmap='prism')
    plt.savefig('test.eps')

if __name__ == "__main__":
    test()
