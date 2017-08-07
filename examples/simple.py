#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import swordfish as sf
import metricplot as mp
import pylab as plt

def main():
    E = np.linspace(0, 11, 100)

    m = np.linspace(2, 10, 20)  # mass (peak position)
    n = np.linspace(1, 10, 20)  # flux strength (normalization)
    I = np.zeros((len(n), len(m), 2, 2))  # Fisher metric (cartesian coordinates)
    UL = np.zeros(len(m))

    for i, m0 in enumerate(m):
        for j, n0 in enumerate(n):
            sigma = 0.7
            flux = sf.func_to_templates(
                    lambda m, n: 1/sigma*n*np.exp(-0.5*(m-E)**2/sigma**2),
                    [m0, n0])
            noise = E**2+10
            exposure = np.ones_like(E)*1.0
            mysf = sf.Swordfish(flux, noise, None, exposure)
            I[j,i] = mysf.fishermatrix()

        flux = sf.func_to_templates(
                lambda n: 1/sigma*n*np.exp(-0.5*(m0-E)**2/sigma**2),
                [1])
        mysf = sf.Swordfish(flux, noise, None, exposure)
        ef = sf.EffectiveCounts(mysf)
        UL[i] = ef.upperlimit(0.95, 0)

    tf = mp.TensorField(m, n, I)
#    tf.quiver()
    vf1, vf2 = tf.get_VectorFields()

    mask = lambda x, y: y>np.interp(x, m, UL)

    lines = vf1.get_streamlines([5, 3], Nmax = 200, mask = mask)
    for line in lines:
        plt.plot(line.T[0], line.T[1], '0.5', lw=1.0)

    lines = vf2.get_streamlines([5, 3], Nmax = 200, mask = mask)
    for line in lines:
        plt.plot(line.T[0], line.T[1], '0.5', lw=1.0)

    contour = tf.get_contour([8, 4.0], 1, Npoints=328)
    plt.plot(contour.T[0], contour.T[1], 'b')
    contour = tf.get_contour([8, 4.0], 2, Npoints=328)
    plt.plot(contour.T[0], contour.T[1], 'b--')
    contour = tf.get_contour([8, 4.0], 3, Npoints=1000)
    plt.plot(contour.T[0], contour.T[1], 'b:')

    contour = tf.get_contour([4, 7.0], 1, Npoints=128)
    plt.plot(contour.T[0], contour.T[1], 'b')
    contour = tf.get_contour([4, 7.0], 2, Npoints=128)
    plt.plot(contour.T[0], contour.T[1], 'b--')
    contour = tf.get_contour([4, 7.0], 3, Npoints=128)
    plt.plot(contour.T[0], contour.T[1], 'b:')

    plt.plot(m, UL, 'r')
    plt.xlim([2, 10])
    plt.ylim([0, 10])

    plt.savefig('test.eps')

if __name__ == "__main__":
    main()
