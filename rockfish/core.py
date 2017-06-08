#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import healpy as hp
import numpy as np
import scipy.sparse.linalg as la

class Fish(object):
    def __init__(self, flux, noise, systematics, exposure):
        self.flux = flux
        self.noise = noise
        self.systematics = systematics
        self.exposure = exposure

    def infomatrix(self, solver = "direct"):
        D = opDiagonal(model.noise/model.exposure) + model.systematics
        n = length(model.flux)
        x = Vector{Vector{Float64}}(n)
        if solver == "direct":
            invD = inv(full(D))
            for i in range(n):
                x[i] = invD*model.flux[i]
        elif solver == "cg":
            for i in range(n):
                Pl = opDiagonal(diag(D))
                x[i] = cg(D, model.flux[i], Pl = Pl, verbose=true, maxiter = 10)
        else:
            error("Solver unknown.")
        I = Matrix{Float64}(n,n)
        for i in range(n):
            for j in range(i):
                tmp = sum(model.flux[i].*x[j])
                I[i,j] = tmp
                I[j,i] = tmp
        return I

    def infoflux(self, solver = "direct", **kwargs):
        D = opDiagonal(model.noise./model.exposure) + model.systematics
        n = length(model.flux)
        x = Vector{Vector{Float64}}(n)
        if solver == "direct":
            invD = inv(full(D))
            for i in range(n):
                x[i] = invD*model.flux[i]
        elif solver == "cg":
            for i in range(n):
                x[i] = cg(D, model.flux[i], verbose=true, maxiter = maxiter)
        else:
            raise KeyError("Solver unknown.")
        I = Matrix{Float64}(n,n)
        F = Matrix{Vector{Float64}}(n,n)
        for i in range(n):
            for j in range(i):
                tmp = x[i].*x[j].*model.noise./model.exposure.^2
                F[i,j] = tmp
                F[j,i] = tmp
                tmp = sum(model.flux[i].*x[j])
                I[i,j] = tmp
                I[j,i] = tmp
        return F, I


