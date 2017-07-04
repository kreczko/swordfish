#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import healpy as hp
import pylab as plt
import scipy.sparse as sp
import inspect
from copy import deepcopy

# Hierarchical Adaptive Resolution Pixelization of the Sphere
# A thin python wrapper around healpix

class HARPix():
    def __init__(self, verbose = False, dims = ()):
        self.ipix = np.empty((0,), dtype=np.int64)
        self.order = np.empty((0,), dtype=np.int8)
        self.dims = dims
        self.data = np.empty((0,)+self.dims, dtype=np.float64)
        self.verbose = verbose

    @classmethod
    def from_healpix(cls, m):
        npix = len(m)
        nside = hp.npix2nside(npix)
        order = hp.nside2order(nside)

        dims = np.shape(m[0])
        r = cls(dims = dims)
        r.data = m
        r.ipix = np.arange(npix, dtype=np.int64)
        r.order = np.ones(npix, dtype=np.int8)*order
        return r

    def print_info(self):
        print "Number of pixels: %i"%len(self.data)
        print "Minimum nside:    %i"%hp.order2nside(min(self.order))
        print "Maximum nside:    %i"%hp.order2nside(max(self.order))
        return self

    def add_singularity(self, vec, r0, r1, n = 100):
        sr0 = np.deg2rad(r0)**2*np.pi/n
        sr1 = np.deg2rad(r1)**2*np.pi/n
        order0 = int(np.log(4*np.pi/12/sr0)/np.log(4))+1
        order1 = int(np.log(4*np.pi/12/sr1)/np.log(4))+1
        for o in range(order1, order0+1):
            r = r1/2**(o-order1)
            nside = hp.order2nside(o)
            self.add_disc(vec, r, nside, clean = False)
        self.clean()
        return self

    def add_ipix(self, ipix, order, clean = True, fill = 0., insert = False):
        if insert:
            self.ipix = np.append(ipix, self.ipix)
            self.order = np.append(order, self.order)
            self.data = np.append(np.ones((len(ipix),)+self.dims)*fill,
                    self.data, axis=0)
        else:
            self.ipix = np.append(self.ipix, ipix)
            self.order = np.append(self.order, order)
            self.data = np.append(self.data,
                    np.ones((len(ipix),)+self.dims)*fill, axis=0)
        if clean:
            self.clean()
        return self

    def add_iso(self, nside = 1, clean = True, fill = 0.):
        order = hp.nside2order(nside)
        npix = hp.nside2npix(nside)
        ipix = np.arange(0, npix)
        self.ipix = np.append(self.ipix, ipix)
        self.data = np.append(self.data,
                np.ones((len(ipix),)+self.dims)*fill, axis=0)
        self.order = np.append(self.order, order*np.ones(len(ipix), dtype=np.int8))
        if self.verbose: print "add_disc:", len(ipix)
        if clean:
            self.clean()
        return self


    def add_disc(self, vec, radius, nside, clean = True, fill = 0.):
        if len(vec) == 2:
            vec = hp.ang2vec(vec[0], vec[1], lonlat=True)
        radius = np.deg2rad(radius)
        order = hp.nside2order(nside)
        ipix = hp._query_disc.query_disc(nside, vec, radius, nest=True)
        self.ipix = np.append(self.ipix, ipix)
        self.data = np.append(self.data,
                np.ones((len(ipix),)+self.dims)*fill, axis=0)
        self.order = np.append(self.order, order*np.ones(len(ipix), dtype=np.int8))
        if self.verbose: print "add_disc:", len(ipix)
        if clean:
            self.clean()
        return self

    def add_polygon(self, vertices, nside, clean = True, fill = 0.):
        order = hp.nside2order(nside)
        ipix = hp._query_disc.query_polygon(nside, vertices, nest=True)
        self.ipix = np.append(self.ipix, ipix)
        self.data = np.append(self.data,
                np.ones((len(ipix),)+self.dims)*fill, axis=0)
        self.order = np.append(self.order, order*np.ones(len(ipix), dtype=np.int8))
        if self.verbose: print "add_polygon:", len(ipix)
        if clean:
            self.clean()
        return self

    def get_trans_matrix(self, nside):
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

    def get_heaplpix(self, nside, idxs = ()):
        M = self.get_trans_matrix(nside)
        if idxs == ():
            return M.dot(self.data)
        elif len(idxs) == 1:
            return M.dot(self.data[:,idxs[0]])
        elif len(idxs) == 2:
            return M.dot(self.data[:,idxs[0], idxs[1]])
        elif len(idxs) == 3:
            return M.dot(self.data[:,idxs[0], idxs[1], idxs[2]])
        else:
            raise NotImplementedError()

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
                spill_data1 = np.repeat(spill_data[~unsubbed], 4, axis=0)
            else:
                clean_ipix1 = np.empty((0,), dtype=np.int64)
                clean_data1 = np.empty((0,)+self.dims, dtype=np.float64)
                spill_ipix1 = np.empty((0,), dtype=np.int64)
                spill_data1 = np.empty((0,)+self.dims, dtype=np.float64)

            unsubbed = np.in1d(self.ipix[mask], sub_ipix, invert = True)
            clean_ipix2 = self.ipix[mask][unsubbed]
            clean_data2 = self.data[mask][unsubbed]
            spill_ipix2 = np.repeat(self.ipix[mask][~unsubbed] << 2, 4)
            spill_ipix2 += np.tile(np.arange(4), int(len(spill_ipix2)/4))
            spill_data2 = np.repeat(self.data[mask][~unsubbed], 4, axis=0)

            clean_ipix_mult = np.append(clean_ipix1, clean_ipix2)
            clean_data_mult = np.append(clean_data1, clean_data2, axis=0)
            clean_ipix_sing, inverse = np.unique(clean_ipix_mult,
                    return_inverse = True)
            clean_data_sing = np.zeros((len(clean_ipix_sing),)+self.dims)
            np.add.at(clean_data_sing, inverse, clean_data_mult)
            clean_ipix.extend(clean_ipix_sing)
            clean_data.extend(clean_data_sing)
            clean_order.extend(np.ones(len(clean_ipix_sing), dtype=np.int8)*o)

            spill_ipix = np.append(spill_ipix1, spill_ipix2)
            spill_data = np.append(spill_data1, spill_data2, axis=0)

        self.ipix = np.array(clean_ipix)
        self.data = np.array(clean_data)
        self.order = np.array(clean_order)
        return self

    def __iadd__(self, other):
        self.data = np.append(self.data, other.data, axis=0)
        self.ipix = np.append(self.ipix, other.ipix)
        self.order = np.append(self.order, other.order)
        self.clean()
        return self

    def __mul__(self, other):
        if isinstance(other, HARPix):
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
        if isinstance(other, HARPix):
            h = deepcopy(self)
            h += other
            return h
        else:
            raise NotImplementedError

    def remove_zeros(self):
        mask = self.data != 0.
        self.ipix = self.ipix[mask]
        self.data = self.data[mask]
        self.order = self.order[mask]
        return self

    def get_area(self):
        """Return area covered by map in steradian."""
        sr = 4*np.pi/12*4.**-self.order
        return sum(sr)

    def get_integral(self):
        """Return area covered by map in steradian."""
        sr = 4*np.pi/12*4.**-self.order
        return sum(sr*self.data)

    def mul_sr(self):
        sr = 4*np.pi/12*4.**-self.order
        self.data *= sr
        return self

    def div_sr(self):
        sr = 4*np.pi/12*4.**-self.order
        self.data /= sr
        return self

    def mul_func(self, func, mode = 'lonlat', **kwargs):
        values = self._evalulate(func, mode = mode, **kwargs)
        self.data *= values
        return self

    def add_func(self, func, mode = 'lonlat', **kwargs):
        values = self._evalulate(func, mode = mode, **kwargs)
        self.data += values
        return self

    def _evalulate(self, func, mode = 'lonlat', center = None):
        nargs = len(inspect.getargspec(func).args)
        signature = "()"
        signature += ",()"*(nargs-1)
        signature += "->"
        if self.dims== ():
            signature += "()" 
        elif len(self.dims) == 1:
            signature += "(n)"
        elif len(self.dims) == 2:
            signature += "(n,m)"
        elif len(self.dims) == 3:
            signature += "(n,m,k)"
        else:
            raise NotImplementedError()
        f = np.vectorize(func, signature = signature)
        if mode == 'lonlat':
            lon, lat = self.get_lonlat()
            values = f(lon, lat)
        elif mode == 'dist':
            dist = self.get_dist(center[0], center[1])
            values = f(dist)
        else:
            raise KeyError("Mode unknown.")
        return values

    def apply_mask(self, mask_func, mode = 'lonlat'):
        self.mul_func(mask_func, mode = mode)
        self.remove_zeros()
        return self

    def get_dist(self, lon, lat):
        lonV, latV = self.get_lonlat()
        dist = hp.rotator.angdist([lon, lat], [lonV, latV], lonlat=True)
        return dist

    def add_random(self):
        self.data += np.random.random(np.shape(self.data))
        return self

    def get_lonlat(self):
        orders = np.unique(self.order)
        lon = np.zeros(len(self.data))
        lat = np.zeros(len(self.data))
        for o in orders:
            nside = hp.order2nside(o)
            mask = self.order == o
            ipix = self.ipix[mask]
            lon[mask], lat[mask] = hp.pix2ang(nside, ipix, nest = True, lonlat = True)
        lon = np.mod(lon+180, 360) - 180
        return lon, lat

def test():
    h = HARPix()

    h.add_iso(nside = 2)
    h.add_singularity((0,0), 0.1, 100, n = 1000)#.add_random()
    h.add_func(lambda d: 0.1/d, center = (0,0), mode='dist')
    h.add_singularity((30,20), 0.1, 100, n = 1000)#.add_random()
    h.add_func(lambda d: 0.1/d, center = (30,20), mode='dist')
    h.add_random()
    #h.apply_mask(lambda l, b: abs(b) < 3)
    h.print_info()

    m = h.get_heaplpix(256)
    hp.mollview(np.log10(m), nest = True, cmap='gnuplot')
    plt.savefig('test.eps')
    # quit()

    npix = hp.nside2npix(8)
    m = np.random.random((npix, 2,3))
    h=HARPix.from_healpix(m)
    m = h.get_heaplpix(128, idxs = (1,1))
    h.print_info()
    hp.mollview(np.log10(m), nest = True, cmap='gnuplot')
    plt.savefig('test2.eps')

    h = HARPix(dims=(10,)).add_iso(fill = 100)
    for i in range(10):
        lonlat = (40*i, 10*i)
        h0 = HARPix(dims=(10,))
        h0.add_peak(lonlat, .01, 10)
        print np.shape(h0.data)
        x = np.linspace(1, 10, 10)
        h0.add_func(lambda dist: x/(dist+0.01), mode = 'dist', center = lonlat)
        h += h0
    m = h.get_heaplpix(128, idxs=(4,))
    h.print_info()
    hp.mollview(np.log10(m), nest = True, cmap='gnuplot', min = 1, max = 4)
    plt.savefig('test3.eps')

if __name__ == "__main__":
    test()
