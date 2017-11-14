#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Overview and motivation
=======================

The purpose of swordfish is to enable fast and informative forecasting for a
broad class of statistical models relevant for praticle physics and astronomy,
without requiring expensive Monte Carlos.  The results are in most cases
accurate, with a few important limitations discussed below.

With swordfish one can calculate

- expected median detection thresholds
- expected median upper limits
- reconstruction contours

However, swordfish allows also to calculate uncommon (but very useful)
quantities like the

- Fisher information matrix
- Fisher information flux
- Equivalent signal and background counts
- Estimate for trials factor

which can guide the optimization of search strategies of experimental design.

The model implemented in swordfish is a Poisson Point Process with constraints
additive components and general background covariance matrix.  This model is
encompasses counting experiments in the zero background regime as well as
scenarios that are completely systematics domainted.

Swordfish can be used in the low-statistics regime as well as parts of the
parameter space that are very degenerate.  The main limitation is that it
cannot handle situations where both happens at the same time.

"""

from __future__ import division
import numpy as np
import scipy.sparse.linalg as la
import scipy.sparse as sp
from scipy import stats
from scipy.special import gammaln
from scipy.linalg import sqrtm
from scipy.optimize import fmin_l_bfgs_b
import copy

import swordfish.metricplot as mp


class Swordfish(object):
    r"""Fisher analysis of general penalized Poisson likelihood models.

    A model is defined by the following quantities

    * $S_i^{(k)}$: 
        Signal components, with $k=1, \dots, n$ components and $i=1,\dots, N$
        bins.
    - $B_i$: Background
    - $E_i$: Exposure
    - $K\_{ij}$: Background covariance matrix
    - $T_i$: Parameter constraints
    
    The corresponding likelihood function is given by
    $$
    \ln\mathcal{L}(\vec d|\vec\theta) = \max\_{\delta B_i} \left[
    \ln\mathcal{L}_p(\vec d| \vec\mu(\vec\theta, \delta \vec B)) +
    \ln\mathcal{L}_c(\vec\theta, \delta\vec B)
    \right]
    $$
    with a Poisson part
    $$
    \ln\mathcal{L}_p(\vec d|\vec \mu) = \sum\_{i=1}^N d_i \cdot\ln
    \mu_i - \mu_i -\ln\Gamma(d_i+1)
    $$
    and a constraint part
    $$
    \ln\mathcal{L}_c(\vec\theta, \delta \vec B) = 
    \frac12 \sum_i \left(\frac{\theta_i}{T_i}\right)^2
    +\frac{1}{2}\sum\_{i,j} \delta B_i \left(K^{-1}\right)\_{ij} \delta B_j 
    \;,
    $$
    where the expected differential number of counts is given by
    $$
    \mu_i(\vec\theta,\delta \vec B) = \left[\sum\_{k=1}^n \theta_k
    S_i^{(k)}+B_i+\delta B_i\right]\cdot E_i \;.
    $$

    Instances of this class define the model parameters, and provide methods to
    calcualte the associated Fisher information matrix and the information
    flux.

    """
    def __init__(self, S, B, E = None, K = None, T = None, solver = 'direct',
            verbose = False):
        r"""Constructor.

        Parameters
        ----------
        * `S` [matrix-like, shape=(n_comp, n_bins)]:
            Flux of signal components.
        * `B` [vector-like, shape=(n_bins)]:
            Background flux.
        * `E` [vector-like, shape=(n_bins), or `None`]:
            Exposure.  The value `None` implies that $E_i=1$.
        * `K` [matrix-like, shape=(n_bins, n_bins), or `None`]:
            Covariance matrix of background flux uncertainties.  Can handle
            anything that can be cast to a `LinearOperator` objects, in
            particular dense and sparse matrices.  The value `None` implies
            $\delta B_i = 0$.
        * `T` [vector-like or list-like, shape=(n_comp), or `None`]:
            Constraints on model parameters.  A list element `None` implies
            that no constraint is applied to the corresponding parameter.  If
            `T=None` no constraints are applied to any parameter.
        * `solver` [`'direct'` or `'cg'`]: 
            Solver method used for matrix inversion, either conjugate gradient
            (cg) or direct matrix inversion.
        * `verbose` [boolean]:
            If set `True` various methods print additional information.
        """
        S = np.array(S, dtype='float64')
        assert S.ndim == 2, "S is not matrix-like."
        n_comp, n_bins = S.shape
        self._flux = S

        B = np.array(B, dtype='float64')
        assert B.shape == (n_bins,), "B has incorrect shape."
        self._noise = B 

        if E is None:
            E = np.ones(n_bins, dtype='float64')
        else:
            E = np.array(E, dtype='float64')
            assert E.shape == (n_bins,), "E has incorrect shape."
        self._exposure = E

        if K is None:
            self._sysflag = False
            K = la.LinearOperator((n_bins, n_bins), 
                    matvec = lambda x: np.zeros_like(x))
        else:
            self._sysflag = True
            assert K.shape == (n_bins, n_bins), "K has incorrect shape."
            K = la.aslinearoperator(K) if K is not None else None
        self._systematics = K

        T = self._get_constraints(T, n_comp)
        self._constraints = T

        self._nbins = n_bins
        self._ncomp = n_comp

        self._verbose = verbose
        self._solver = solver
        self._scale = self._get_auto_scale(S, E)
        self._cache = None  # Initialize internal cache

    @staticmethod
    def _get_auto_scale(flux, exposure):
        return np.array(
                [1./(f*exposure).max() for f in flux]
                )

    @staticmethod
    def _get_constraints(constraints, ncomp):
        assert constraints is None or len(constraints) == ncomp
        if constraints is not None:
            constraints = np.array(
                [np.inf if x is None or x == np.inf else x for x in
                    constraints], dtype = 'float64')
            if any(constraints<=0.):
                raise ValueError("Constraints must be positive or None.")
        else:
            constraints = np.ones(ncomp)*np.inf
        return constraints

    def _summedNoise(self, theta = None):
        noise_tot = self._noise*1.  # Make copy
        if theta is not None: 
            for i in range(max(self._ncomp, len(theta))):
                noise_tot += theta[i]*self._flux[i]
        return noise_tot

    def mu(self, theta = None):
        r"""Return expectation values for given model parameters.

        Parameters
        ----------
        * `theta` [vector-like, shape=(n_comp), or `None`]:
            Model parameters.  The value `None` implies $\theta_i = 0$.

        Returns
        -------
        * `mu` [vector-like, shape=(n_bins)]:
            Expectation value, $\mu_i(\vec \theta, \delta \vec B = 0)$.
        """
        return self._summedNoise(theta)*self._exposure

    def _solveD(self, theta = None):
        """
        Calculates:
            N = noise + theta*flux
            D = diag(E)*Sigma*diag(E)+diag(N*E)
            x[i] = D^-1 flux[i]*E

        Note: if Sigma = None: x[i] = flux[i]/noise

        Returns:
            x, noise, exposure
        """
        noise = self._summedNoise(theta)
        exposure = self._exposure
        spexp = la.aslinearoperator(sp.diags(exposure))
        D = (
                la.aslinearoperator(sp.diags(noise*exposure))
                + spexp*self._systematics*spexp
                )
        x = np.zeros((self._ncomp, self._nbins))
        if not self._sysflag:
            for i in range(self._ncomp):
                x[i] = self._flux[i]/noise*exposure
        elif self._sysflag and self._solver == "direct":
            dense = D(np.eye(self._nbins))
            invD = np.linalg.linalg.inv(dense)
            for i in range(self._ncomp):
                x[i] = np.dot(invD, self._flux[i]*exposure)*exposure
        elif self._sysflag and self._solver == "cg":
            def callback(x):
                if self._verbose:
                    print len(x), sum(x), np.mean(x)
            for i in range(self._ncomp):
                x0 = self._flux[i]/noise if self._cache is None else self._cache/exposure
                x[i] = la.cg(D, self._flux[i]*exposure, x0 = x0, callback = callback, tol = 1e-3)[0]
                x[i] *= exposure
                self._cache= x[i]
        else:
            raise KeyError("Solver unknown.")
        return x, noise, exposure

    def fishermatrix(self, theta = None):
        r"""Return Fisher information matrix.

        Parameters
        ----------
        * `theta` [vector-like, shape=(n_comp)]:
            Model parameters.  The value `None` is equivalent to
            $\vec\theta=0$.

        Returns
        -------
        * `I` [matrix-like, shape=(n_comp, n_comp)]:
            Fisher information matrix, $\mathcal{I}\_{ij}$.
        """
        x, noise, exposure = self._solveD(theta=theta)
        I = np.zeros((self._ncomp,self._ncomp))
        for i in range(self._ncomp):
            for j in range(i+1):
                tmp = sum(self._flux[i]*x[j])
                I[i,j] = tmp
                I[j,i] = tmp
        return I+np.diag(1./self._constraints**2)

    def infoflux(self, theta = None):
        r"""Return Fisher information flux.

        Parameters
        ----------
        * `theta` [vector-like, shape=(n_comp)]:
            Model parameters.  The value `None` is equivalent to
            $\vec\theta=0$.

        Returns
        -------
        * `F` [3-D array, shape=(n_comp, n_comp, n_bins)]:
            Fisher information flux.
        """
        x, noise, exposure = self._solveD(theta=theta)

        F = np.zeros((self._ncomp,self._ncomp,self._nbins))
        for i in range(self._ncomp):
            for j in range(i+1):
                tmp = x[i]*x[j]*noise/(exposure**2.)
                F[i,j] = tmp
                F[j,i] = tmp
        return F

    def effectivefishermatrix(self, indexlist, **kwargs):
        """Return effective Fisher information matrix for subset of components.

        Parameters
        ----------
        * `indexlist` [integer, or list of integers]:
            Components of interest.
        * `**kwargs`:
            Passed on to call of `fishermatrix`.

        Returns
        -------
        * `Ieff` [float, or matrix-like, shape=(len(indexlist),
            len(indexlist))]:
            Effective Fisher information matrix.

        """
        if isinstance(indexlist, np.int):
            indexlist = [indexlist]
        indices = np.setdiff1d(range(self._ncomp), indexlist)
        n = len(indexlist)

        I = self.fishermatrix(**kwargs)
        A = I[indexlist,:][:,indexlist]
        B = I[indices,:][:,indexlist]
        C = I[indices,:][:,indices]
        invC = np.linalg.linalg.inv(C)

        Ieff = A - B.T.dot(invC.dot(B))

        return Ieff

    def variance(self, i, **kwargs):
        r"""Return variance of $\theta_i$.

        Parameters
        ----------
        * `i` [integer]: 
            Index of component of interest
        * `**kwargs`:
            Passed on to call of `fishermatrix`.

        Returns
        -------
        * `var` [float]:
            Variance of parameter $\theta_i$.
        """
        I = self.fishermatrix(**kwargs)
        invI = np.linalg.linalg.inv(I)
        return invI[i,i]

    def effectiveinfoflux(self, i, **kwargs):
        """Return effective Fisher Information Flux.

        Parameters
        ----------
        * `i` [integer]:
            Index of component of interest.
        * `**kwargs`:
            Passed on to call of `fishermatrix` and `infoflux`.

        Returns
        -------
        * `Feff` [vector-like, shape=(n_bins)]:
            Effective information flux.
        """
        F = self.infoflux(**kwargs)
        I = self.fishermatrix(**kwargs)
        n = self._ncomp
        if n == 1:
            return F[i,i]
        indices = np.setdiff1d(range(n), i)
        eff_F = F[i,i]
        C = np.zeros(n-1)
        B = np.zeros((n-1,n-1))
        for j in range(n-1):
            C[j] = I[indices[j],i]
            for k in range(n-1):
                B[j,k] = I[indices[j], indices[k]]
        invB = np.linalg.linalg.inv(B)
        for j in range(n-1):
            for k in range(n-1):
                eff_F = eff_F - F[i,indices[j]]*invB[j,k]*C[k]
                eff_F = eff_F - C[j]*invB[j,k]*F[indices[k],i]
                for l in range(n-1):
                    for m in range(n-1):
                        eff_F = eff_F + C[j]*invB[j,l]*F[indices[l],indices[m]]*invB[m,k]*C[k]
        return eff_F

    # Likelihood with systematics
    def lnL(self, theta, theta0, deltaB = None, epsilon = 1e-3, derivative =
            False, mu_overwrite = None):
        r"""Return log-likelihood function, assuming Asimov data.

        Parameters
        ----------
        * `theta` [vector-like, shape=(n_comp)]:
            Model parameters.
        * `theta0` [vector-like, shape=(n_comp)]:
            Model parameters used for calculation of Asimov data.  Note that
            Asimov data is generated assuming $\delta\vec B =0$.
        * `deltaB`: [vector-like, shape=(n_bins)]:
            Background variations of model.  A value of `None` corresponds to
            $\delta\vec B = 0$.
        * `epsilon` [float]:
            Fraction of diagonal noise that is added to K to stabilize matrix
            inversion.  Must be small enough to not affect results.
        * `derivative` [boolean]:
            If set to `True`, function also resturns partial derivatives w.r.t.
            model parameters.
        * `mu_overwrite` [vector-like, shape=(n_bins)]:
            This parameter is internally used to handle non-linear models.  It
            overwrites the model predictions that are derived from `theta` and
            `deltaB`.  In that case, `theta` and `deltaB` only affect the
            constraint part of the likelihood.

        Returns
        -------
        * `lnL` [float]:
            Log-likelihood, including Poisson and constraint part.

        OR if `derivative == True`

        * `lnL, dlnL_dtheta, dlnL_deltaB` 
            [(float, vector-like with shape=(n_comp), vector-like with shape=(n_bins))]:
            Log-likelihood and its partial derivatives.
        """
        dmu = deltaB
        theta = np.array(theta, dtype='float64')
        theta0 = np.array(theta0, dtype='float64')
        mu0 = self._summedNoise(theta0)*self._exposure
        systnoise = self._summedNoise(theta0)*epsilon/self._exposure
        if mu_overwrite is None:
            mu =  self._summedNoise(theta)*self._exposure
        else:
            mu = mu_overwrite.copy()
        if dmu is None:
            dmu = np.zeros_like(self._exposure)
        if self._sysflag:
            mu += dmu*self._exposure
        self._at_bound = any(mu<mu0*1e-6)
        mu = np.where(mu<mu0*1e-6, mu0*1e-6, mu)
        lnL = (mu0*np.log(mu)-mu-0*gammaln(mu0+1)).sum()
        #print mu0.sum(), mu.sum()
        lnL -= (0.5*theta**2/self._constraints**2).sum()
        if self._sysflag:
            dense = self._systematics(np.eye(self._nbins))
            #invS = np.linalg.linalg.inv(dense+np.eye(self._nbins)*epsilon)
            invS = np.linalg.linalg.inv(dense+np.diag(systnoise))
            lnL -= 0.5*(invS.dot(dmu)*dmu).sum()
        if derivative:
            dlnL_dtheta = (mu0/mu*self._flux*self._exposure-self._flux*self._exposure).sum(axis=1)
            dlnL_dtheta -= theta/self._constraints**2
            if self._sysflag:
                dlnL_dmu = mu0/mu*self._exposure - self._exposure - invS.dot(dmu)
            else:
                dlnL_dmu = None
            return lnL, dlnL_dtheta, dlnL_dmu
        else:
            return lnL

    def profile_lnL(self, theta, theta0, epsilon = 1e-3, free_theta = None,
            mu_overwrite = None):
        r"""Return profile log-likelihood.

        Note: Profiling is done using `scipy.optimize.fmin_l_bfgs_b`.  All
        $\delta \vec B$ are treated as free parameters, as well as those model
        parameters $\theta_i$ that are specified in `free_theta`.

        Parameters
        ---------
        * `theta` [vector-like, shape=(n_comp)]:
            Model parameters.
        * `theta0` [vector-like, shape=(n_comp)]:
            Model parameters used for calculation of Asimov data.  Note that
            Asimov data is generated assuming $\delta\vec B =0$.
        * `epsilon` [float]:
            Fraction of diagonal noise that is added to K to stabilize matrix
            inversion.  Must be small enough to not affect results.
        * `free_theta` [list-like, shape=(n_comp)]:
            If a list entry is set to `True`, the corresponding model parameter
            will be maximized.
        * `mu_overwrite` [vector-like, shape=(n_bins)]:
            See `lnL`.

        Returns
        -------
        * `profile_lnL` [float]:
            Profile log-likelihood.
        """
        theta = np.array(theta, dtype='float64')
        theta0 = np.array(theta0, dtype='float64')
        if free_theta is None:
            free_theta = np.zeros(len(theta), dtype='bool')
        else:
            free_theta = np.array(free_theta, dtype='bool')
        Nfree_theta = (free_theta).sum()
        Nsyst = len(self._exposure) if self._sysflag else 0
        N = Nfree_theta + Nsyst
        x0 = np.zeros(N)
        x0[:Nfree_theta] = theta[free_theta]
        fp = 0.
        def f(x):
            global fp
            theta[free_theta] = x[:Nfree_theta]
            dmu = x[Nfree_theta:]
            lnL, grad_theta, grad_dmu = self.lnL(theta, theta0, deltaB = dmu,
                    epsilon = epsilon, derivative = True, mu_overwrite = mu_overwrite)
            fp = np.zeros(N)
            fp[:Nfree_theta] = grad_theta[free_theta]
            fp[Nfree_theta:] = grad_dmu
            return -lnL
        def fprime(x):
            global fp
            return -fp
        if N == 0.:
            return self.lnL(theta, theta0, mu_overwrite = mu_overwrite)
        result = fmin_l_bfgs_b(f, x0, fprime, approx_grad = False)
        if self._verbose:
            print "Best-fit parameters:", result[0]
        if self._at_bound:
            print "WARNING: No maximum with non-negative flux found."
            return -result[1]
        else:
            return -result[1]

class EuclideanizedSignal(object):
    # WARNING: This only cares about the covariance matrix and background, not
    # the individual S components
    def __init__(self, model):
        """*Effective vector* calculation based on a `Swordfish` instance.

        The distance method provides a simple way to calculate the
        expected statistical distance between two signals, accounting for
        background variations and statistical uncertainties.  In practice, it
        maps signal spectra on distance vectors such that the Eucledian
        distance between vectors corresponds to the statistical
        difference between signal spectra.
        """
        self._model = copy.deepcopy(model)
        self._A0 = None

    def _get_A(self, S = None):
        """Return matrix `A`, such that x = A*S is the Eucledian distance vector."""
        noise = self._model._noise.copy()*1.  # Noise without anything extra
        if S is not None:
            noise += S
        exposure = self._model._exposure
        spexp = la.aslinearoperator(sp.diags(exposure))

        # Definition: D = N*E + E*Sigma*E
        D = (
                la.aslinearoperator(sp.diags(noise*exposure))
                + spexp*self._model._systematics*spexp
                )
        # TODO: Add all components and their errors to D

        D = D(np.eye(self._model._nbins))  # transform to dense matrix
        invD = np.linalg.linalg.inv(D)
        A2 = np.diag(exposure).dot(invD).dot(np.diag(exposure))
        A = sqrtm(A2)
        Kdiag = np.diag(self._model._systematics.dot(np.eye(self._model._nbins)))
        nA = np.diag(np.sqrt(noise*exposure+Kdiag*exposure**2)).dot(A)
        return nA

    def x(self, i, S):
        """Return model distance vector.

        Parameters
        ----------
        * `i` [integer]:
            Index of component of interest.
        * `S` [array-like, shape=(n_bins)]:
            Flux of signal component
        """
        A = self._get_A(S)
        return A.dot(S)

class EquivalentCounts(object):
    """*Equivalent counts* analysis based on a `Swordfish` instance.

    The equivalent counts method can be used to derive
    
    - expected upper limits
    - discovery reach
    - equivalent signal and background counts
    
    based on the Fisher information matrix of general penalized Poisson
    likelihood models.  The results are usually rather accurate, and work in
    the statistics limited, background limited and systematics limited regime.
    However, they require that the parameter of interest is (a) unconstrained
    and (b) corresponds to a component normalization.
    """

    def __init__(self, model):
        """Constructor.

        Paramters
        ---------
        * `model` [instance of `Swordfish` class]:
            Input Swordfish model.
        """
        self._model = model

    def totalcounts(self, i, theta_i):
        """Return total signal and background counts.

        Parameters
        ----------
        * `i` [integer]:
            Index of component of interest.
        * `theta_i` [float]:
            Normalization of component i.

        Returns
        -------
        * `s` [float]:
            Total signal counts.
        * `b` [float]:
            Total background counts.
        """
        s = sum(self._model._flux[i]*self._model._exposure*theta_i)
        b = sum(self._model._noise*self._model._exposure)
        return s, b

    def equivalentcounts(self, i, theta_i):
        """Return equivalent signal and background counts.

        Parameters
        ----------
        * `i` [integer]:
            Index of component of interest.
        * `theta_i` [float]:
            Normalization of component i.

        Returns
        -------
        * `s` [float]:
            Equivalent signal counts.
        * `b` [float]:
            Equivalent background counts.
        """
        I0 = 1./self._model.variance(i)
        thetas = np.zeros(self._model._ncomp)
        thetas[i] = theta_i
        I = 1./self._model.variance(i, theta = thetas)
        if I0 == I:
            return 0., None
        seff = 1/(1/I-1/I0)*theta_i**2
        beff = 1/I0/(1/I-1/I0)**2*theta_i**2
        return seff, beff

    def upperlimit(self, i, alpha = 0.05, force_gaussian = False):
        r"""Return expected upper limit.

        Parameters
        ----------
        * `i` [integer]:
            Index of component of interest.
        * `alpha` [float]:
            Statistical significance.  For example, 0.05 means 95% confidence
            level.
        * `force_gaussian` [boolean]:
            Force calculation of Gaussian errors (faster, but only accurate in
            Gaussian regime).

        Returns
        -------
        * `thetaUL` [float]:
            Expected median upper limit on $\theta_i$.
        """
        Z = stats.norm.isf(alpha)
        I0 = 1./self._model.variance(i)
        if force_gaussian:
            return Z/np.sqrt(I0)
        else:
            thetas = np.zeros(self._model._ncomp)
            thetaUL_est = Z/np.sqrt(I0)  # Gaussian estimate
            thetas[i] = thetaUL_est
            I = 1./self._model.variance(i, theta = thetas)
            if (I0-I)<0.02*I:  # 1% accuracy of limits
                thetaUL = thetaUL_est
            else:
                z_list = []
                theta_list = [thetaUL_est]
                while True:
                    theta = theta_list[-1]
                    s, b = self.equivalentcounts(i, theta)
                    if s == 0: b = 1.
                    z_list.append(s/np.sqrt(s+b))
                    if z_list[-1] > Z:
                        break
                    else:
                        theta_list.append(theta*1.3)
                    #print theta, z_list
                thetaUL = np.interp(Z, z_list, theta_list)
            return thetaUL

    def discoveryreach(self, i, alpha = 1e-6, force_gaussian = False):
        r"""Return expected discovery reach.

        Parameters
        ----------
        * `i` [integer]:
            Index of component of interest.
        * `alpha` [float]:
            Statistical significance.
        * `force_gaussian` [boolean]:
            Force calculation of Gaussian errors (faster, but only accurate in
            Gaussian regime).

        Returns
        -------
        * `thetaDT` [float]:
            Expected median discovery reach for $\theta_i$.
        """
        Z = stats.norm.isf(alpha)
        var0 = self._model.variance(i)
        if force_gaussian:  # Gaussian approximation
            return Z*var0**0.5

        thetas = np.zeros(self._model._ncomp)
        thetaDT_est = Z*np.sqrt(var0)  # Gaussian approx. as starting point
        thetas[i] = thetaDT_est
        var = self._model.variance(i, theta = thetas)
        if abs(var0 - var) < 0.02*var:  # Still Gaussian enough
            return Z*var0**0.5
        z_list = []
        theta_list = [thetaDT_est]
        while True:
            theta = theta_list[-1]
            s, b = self.equivalentcounts(i, theta)
            if s == 0: b = 1.
            z_list.append((s+b)*np.log((s+b)/b)-s)
            if z_list[-1] > Z**2/2:
                break
            else:
                theta_list.append(theta*1.3)
        return np.interp(Z**2/2, z_list, theta_list)

def _func_to_templates(flux, x, dx = None):
    """Return finite differences for use in SwordFish."""
    x = np.array(x, dtype='float64')
    if dx is None:
        dx = x*0.01+0.001
    fluxes = []
    #fluxes.append(flux(*x))
    for i in range(len(x)):
        xU = copy.copy(x)
        xL = copy.copy(x)
        xU[i] += dx[i]
        xL[i] -= dx[i]
        df = (flux(*xU) - flux(*xL))/2./dx[i]
        fluxes.append(df)
    return fluxes


class Funkfish(object):
    r"""`Swordfish`, `EquivalentCounts` and `iminuit`-factory, based on non-linear models.

    The underlying likelihood function is identical to the one of `Swordfish`,
    with the only difference that model expectations are derived from
    $$
    \mu_i(\vec\theta, \delta \vec B) = \left[ f_i(\vec\theta) + \delta B_i\right]\cdot E_i
    $$

    `Funkfish` can generate `Swordfish` and `EquivalentCounts` objects as local
    approximations to the non-linear model, as well as `iminuit` instances
    based on the non-linear model directly.  This facilitates (a) the fast
    analysis of non-linear models and (b) the comparison between results based
    on Fisher information and results from a full likelihood analysis.
    """

    def __init__(self, f, theta0, E = None, K = None, T = None):
        r"""Constructor.

        Parameters
        ----------
        * `f` [function]:
            Function of `n_comp` float arguments, returns `n_bins` flux
            expectation values, $\vec\mu(\vec\theta)$.
        * `theta0` [vector-like, shape=(n_comp)]:
            Default model parameters.
        * `E` [vector-like, shape=(n_bins)]:
            Exposure.  See `Swordfish` documentation for details.
        * `K` [matrix-like]:
            Covariance matrix.  See `Swordfish` documentation for details.
        * `T` [vector-like]:
            Model parameter constraints.  See `Swordfish` documentation for
            details.
        """
        self._f = f
        self._x0 = np.array(theta0, dtype='float64')
        self._Sigma = K
        self._exposure = E
        self._constraints = T

    def _get_x0(self, x):
        """Get updated x0."""
        if isinstance(x, dict):
            x0 = self._x0.copy()
            for i in x:
                x0[i] = x[i]
            return x0
        elif x is not None:
            return x
        else:
            return self._x0

    @staticmethod
    def _init_minuit(f, x = None, x_fix = None, x_err = None, x_lim = None, errordef = 1, **kwargs):
        """Initialize minuit using no-nonsense interface."""
        try:
            import iminuit
        except ImportError:
            raise ImportError(
                    "This function requires that the module iminuit is installed.")

        N = len(x)
        if x_err is not None:
            assert len(x_err) == N
        if x_lim is not None:
            assert len(x_lim) == N
        if x_fix is not None:
            assert len(x_fix) == N
        varnames = ["x"+str(i) for i in range(N)]
        def wf(*args):
            x = np.array(args)
            return f(x)
        for i, var in enumerate(varnames):
            kwargs[var] = x[i]
            if x_lim is not None:
                kwargs["limit_"+var] = x_lim[i]
            if x_err is not None:
                kwargs["error_"+var] = x_err[i]
            if x_fix is not None:
                kwargs["fix_"+var] = x_fix[i]
        return iminuit.Minuit(wf, forced_parameters = varnames, errordef =
                errordef, **kwargs)

    def Swordfish(self, theta0 = None):
        r"""Generate `Swordfish` instance.

        The generated `Swordfish` instance is a local approximation to the
        non-linear model.  It is defined by
        $$
        B_i \equiv f_i(\vec\theta_0)
        $$
        and
        $$
        S_i \equiv \frac{\partial f_i}{\partial \theta_i}(\vec\theta_0)
        $$
        where $\vec\theta_0$ is the expansion parameter defined below.

        Parameters
        ----------
        * `theta0` [dictionary, or vector-like with shape=(n_comp), or `None`]:
            If vector-like, it overwrites the default `theta0`.  A dictionary with keys
            $i$ and values $\theta_i$ updates the default.  If `None` the
            default is used.

        Returns
        -------
        * `SF` [`Swordfish` instance]
        """
        x0 = self._get_x0(theta0)
        flux = _func_to_templates(self._f, x0)
        noise = self._f(*x0)
        return Swordfish(flux, noise, self._exposure, self._Sigma, T = self._constraints)

    def EquivalentCounts(self, theta0 = None):
        """Generate `EquivalentCounts` instance.

        Directly generates `EquivalentCounts` instance from `Swordfish`
        instance.  See documentation of `get_Swordfish` method .
        """
        SF = self.Swordfish(theta0)
        return EquivalentCounts(SF)

    def TensorField(self, ix, iy, x_values, y_values, theta0 = None):
        """Generate `TensorField` instance.

        Samples Fisher information matrix on a 2-D grid, and generates an
        instance of `TensorField` that can be used to study parameter
        degeneracies, confidence contours, etc.

        Parameters
        ----------
        * `ix` [integer]:
            Index of first component of interest, mapped on x-axis.
        * `iy` [integer]:
            Index of second component of interest, mapped on y-axis.
        * `x_values` [vector-like, shape=(nx)]:
            Values of model parameter `ix` along x-axis.
        * `y_values` [vector-like, shape=(ny)]:
            Values of model parameter `iy` along y-axis.
        * `theta0` [dictionary, or vector-like with shape=(n_comp), or `None`]:
            If vector-like, it overwrites the default `theta0`.  A dictionary with keys
            $i$ and values $\theta_i$ updates the default.  If `None` the
            default is used.  Defines all parameters, except the parameters
            with indices `ix` and `iy`.

        Returns
        -------
        * `TF` [`TensorField` instance]:
            Based on an interpolated version of the Fisher information matrix
            that is sampled from the (nx, ny) grid defined by `x_values` and
            `y_values`.
        """
        theta0 = self._get_x0(theta0)
        g = np.zeros((len(y_values), len(x_values), 2, 2))
        for i, y in enumerate(y_values):
            for j, x in enumerate(x_values):
                theta0[ix] = x
                theta0[iy] = y
                SF = self.Swordfish(theta0)
                g[i, j] = SF.effectivefishermatrix((ix, iy))
        return mp.TensorField(x_values, y_values, g)

    def iminuit(self, theta0 = None, **kwargs):
        """Return an `iminuit` instance.

        Model parameters are mapped on `iminuit` variables `x0`, `x1`, `x2`, etc

        Parameters
        ----------
        * `theta0`:
            Asimov data
        * `**kwargs*:
            Arguments passed on to `iminuit` constructor.

        Returns
        -------
        * `M` [`iminuit` instance]
        """
        x0 = theta0
        x0 = self._get_x0(x0)  # make list
        SF0 = self.Swordfish(x0)
        def chi2(x):
            mu = self.Swordfish(x).mu()  # Get proper model prediction
            lnL = SF0.profile_lnL(x-x0, x0*0., mu_overwrite = mu)
            return -2*lnL
        x0 = np.array(x0)
        x0err = np.where(x0>0., x0*0.01, 0.01)
        M = self._init_minuit(chi2, x = x0, x_err = x0err, **kwargs)
        return M

class Fishpy(object):
    """Signal ."""
    def __init__(self, B, N = None, T = None, E = None, K = None):
        """Constructor.
        
        Parameters
        ----------
        * `B` [list of equal-shaped arrays with length `n_comp`]:
          Background model
        * `N` [list of non-negative floats with length `n_comp`, or None]:
          Normalization of background components, if `None` assumed to be one.
        * `T` [list of non-negative floats with length `n_comp`, or None]:
          Uncertainty of background components.  In standard deviations.  If
          `None`, all components are assumed to be fixed.
        * `E` [array with the same shape as the components of `B`]:
          Exposure.  If `None`, this is set to one for all bins.
        * `K` [matrix-like]:
          Covariance matrix, meant to refer to the flattened version of the
          background components.  If `None`, it is set to zero.
        """
        if not isinstance(B, list):
            B = [np.array(B, dtype='flota64'),]
        else:
            B = [np.array(b, dtype='float64') for b in B]
            if len(set([b.shape for b in B])) != 1:
                raise ValueError("Incompatible shapes in B.")

        # Save shape, and flatten arrays
        shape = B[0].shape
        B = [b.flatten() for b in B]
        nbins = len(B[0])

        if T is None:
            T = list(np.zeros(len(B), dtype='float64'))
        elif not isinstance(T, list):
            T = [float(T),]
        else:
            T = list(np.array(T, dtype='float64'))

        if len(T) != len(B):
            raise ValueError("T and B must have same length, or T must be None.")

        if K is not None:
            assert K.shape == (nbins, nbins)

        if E is None:
            E = np.ones(nbins)
        else:
            E = np.array(E, dtype='float64').flatten()

        if N is None:
            self._Btot = sum(B)
        else:
            self._Btot = sum([B[i]*N[i] for i in range(len(B))])
        self._B = B  # List of equal-sized arrays
        self._T = T  # List of standard deviations (0., finite or None)
        self._K = la.aslinearoperator(K) if K is not None else None
        self._E = E  # Exposure
        self._shape = shape

    def _ff_factory(self, Sfunc, theta0):
        """Generate Funkfish object.

        Parameters
        ----------
        * `Sfunc` [function]:
          Signal components.
        * `theta0` [vector-like, shape=(n_comp)]:
            Model parameters used for calculation of Asimov data.
        """
        Btot = self._Btot

        K = self._K
        KB = self._B2K(self._B, self._T)
        Ktot = K if KB is None else (KB if K is None else KB+K)
        SfuncB = lambda *args: Sfunc(*args) + Btot
        return Funkfish(SfuncB, theta0, E = self._E, K = Ktot)

    def _sf_factory(self, S, K_only = False, extraB = None):
        """Generate Swordfish object.

        Parameters
        ----------
        * `S` [array or list of arrays]:
          Signal components.
        * `K_only' [boolean]:
          If `True`, dump all background components into `K`.
        """
        if isinstance(S, list):
            S = [np.array(s, dtype='float64') for s in S]
            assert len(set([s.shape for s in S])) == 1.
            assert S[0].shape == self._shape
        else:
            S = [np.array(S, dtype='float64')]
        assert S[0].shape == self._shape
        S = [s.flatten() for s in S]

        Ssf = []  # List of "signal" components for Swordfish
        Tsf = []  # Signal component constraints

        Bsf = copy.deepcopy(self._Btot)
        if extraB is not None:
            Bsf += extraB

        # Collect signal components 
        for s in S:
            Ssf.append(s)
            Tsf.append(None)  # Signals are unconstrained

        # If K-matrix is set anyway, dump everything there for efficiency reasons.
        K = self._K
        if K is not None or K_only:
            KB = self._B2K(self._B, self._T)
            Ktot = K if KB is None else (KB if K is None else KB+K)
        else:
            for i, t in enumerate(self._T):
                if t > 0.:
                    Ssf.append(self._B[i])
                    Tsf.append(self._T[i])
            Ktot = K

        return Swordfish(Ssf, Bsf, E = self._E, K = Ktot, T = Tsf), len(S)

    def _B2K(self, B, T):
        "Transform B and T into contributions to covariance matrix"
        K = None
        for i, t in enumerate(T):
            if t > 0.:
                n_bins = len(B[i])
                Kp = la.LinearOperator((n_bins, n_bins), 
                        matvec = lambda x, B = B[i].flatten(), T = T[i]:
                        B * (x.T*B).sum() * T**2)
                K = Kp if K is None else K + Kp
                # NOTE 1: x.T instead of x is required to make operator
                # work for input with shape (nbins, 1), which can happen
                # internally when transforming to dense matrices.
                # NOTE 2: Thanks to Pythons late binding, _B and _T have to
                # be communicated via arguments with default values.
        return K

    def fishermatrix(self, S, S0 = None):
        """Return Fisher Information Matrix for signal components.

        Parameters
        ----------
        * `S` [list of equal-shaped arrays with length `n_comp`]:
          Signal components.

        Returns
        -------
        * `I` [matrix-like, shape=`(n_comp, n_comp)`]:
          Fisher information matrix.
        """
        # TODO: Extend to function, allow to add to background
        SF, n = self._sf_factory(S, extraB = S0)
        return SF.effectivefishermatrix(range(n), theta = None)

    def covariance(self, S, S0 = None):
        """Return covariance matrix for signal components.

        The covariance matrix is here approximated by the inverse of the Fisher
        information matrix.

        Parameters
        ----------
        * `S` [list of equal-shaped arrays with length `n_comp`]:
          Signal components.

        Returns
        -------
        * `Sigma` [matrix-like, shape=`(n_comp, n_comp)`]:
          Covariance matrix.
        """
        I = self.fishermatrix(S, S0 = S0)
        return np.linalg.linalg.inv(I)

    def infoflux(self, S, S0 = None):
        """Return Fisher information flux.

        Parameters
        ----------
        * `S` [signal arrays]:
          Single signal component.

        Returns
        -------
        * `F` [array like]:
          Fisher information flux.
        """
        SF, n = self._sf_factory(S, extraB = S0)
        assert n == 1
        F = SF.effectiveinfoflux(0, theta = None)
        return np.reshape(F, self._shape)

    def variance(self, S, S0 = None):
        """Return Variance of single signal component.

        Parameters
        ----------
        * `S` [signal arrays]:
          Single signal component.

        Returns
        -------
        * `var` [float]:
          Variance of signal `S`.
        """
        SF, n = self._sf_factory(S, extraB = S0)
        assert n == 1
        return SF.variance(0, theta = None)

    def totalcounts(self, S):  # 1-dim
        """Return total counts.

        Parameters
        ----------
        * `S` [signal arrays]:
          Single signal component.

        Returns
        -------
        * `s` [float]:
          Total signal counts.
        * `b` [float]:
          Total background counts.
        """
        SF, n = self._sf_factory(S)
        assert n == 1
        EC = EquivalentCounts(SF)
        return EC.totalcounts(0, 1.)

    def equivalentcounts(self, S):  # 1-dim
        """Return total counts.

        Parameters
        ----------
        * `S` [signal arrays]:
          Single signal component.

        Returns
        -------
        * `s` [float]:
          Equivalent signal counts.
        * `b` [float]:
          Equivalent background counts.
        """
        SF, n = self._sf_factory(S)
        assert n == 1
        EC = EquivalentCounts(SF)
        return EC.equivalentcounts(0, 1.)

    def upperlimit(self, S, alpha, force_gaussian = False):  # 1-dim
        """Derive projected upper limit.

        Parameters
        ----------
        * `S` [signal arrays]:
          Single signal component.
        * `alpha` [float]:
          Significance level.
        * `force_gaussian` [boolean]:
          Force calculation of Gaussian errors (faster, but use with care).

        Returns
        -------
        * `theta` [float]:
          Normalization of `S` that corresponds to upper limit with
          significance level `alpha`.
        """
        SF, n = self._sf_factory(S)
        assert n == 1
        EC = EquivalentCounts(SF)
        return EC.upperlimit(0, alpha, force_gaussian = force_gaussian)

    @staticmethod
    def _lnP(c, mu):
        # log-Poisson likelihood
        c = c+1e-10  # stablize result
        return (c-mu)+c*np.log(mu/c)

    def significance(self, S):
        """Calculate signal significance.

        Parameters
        ----------
        * `S` [signal arrays]:
          Single signal component.

        Returns
        -------
        * `alpha` [float]:
          Significance of signal.
        """
        s, b = self.equivalentcounts(S)
        Z = np.sqrt(2*(self._lnP(s+b, s+b) - self._lnP(s+b, b)))
        alpha = stats.norm.sf(Z)
        return alpha

    def discoveryreach(self, S, alpha, force_gaussian = False):  # 1-dim
        """Derive discovery reach.

        Parameters
        ----------
        * `S` [signal arrays]:
          Single signal component.
        * `alpha` [float]:
          Significance level.
        * `force_gaussian` [boolean]:
          Force calculation of Gaussian errors (faster, but use with care).

        Returns
        -------
        * `theta` [float]:
          Normalization of `S` that corresponds to discovery with significance
          level `alpha`.
        """
        SF, n = self._sf_factory(S)
        assert n == 1
        EC = EquivalentCounts(SF)
        return EC.discoveryreach(0, alpha, force_gaussian = force_gaussian)

    def equivalentshapes(self, S):  # 1-dim
        """Derive equivalent signal and background shapes.

        Parameters
        ----------
        * `S` [signal arrays]:
          Single signal component.

        Returns
        -------
        * `eqS` [signal array]:
          Equivalent signal.
        * `eqB` [background array]:
          Equivalent noise.
        """
        SF, n  = self._sf_factory(S, K_only = True)
        assert n == 1
        ED = EuclideanizedSignal(SF)
        Kdiag = np.diag(SF._systematics.dot(np.eye(SF._nbins)))
        N = (SF._noise+S)*SF._exposure + Kdiag*SF._exposure**2
        eS = ED.x(0, S)
        return eS, N

    def lnL(self, S, S0):
        """Profile log-likelihood.

        Paramters
        ---------
        * `S` [signal arrays]:
          Single signal component (model prediction).
        * `S0` [signal arrays]:
          Single signal component (mock data).

        Returns
        -------
        * `lnL` [float]:
          Profile log-likelihood.
        """
        SF, n  = self._sf_factory(S, K_only = True)  # Model point
        SF0, n0  = self._sf_factory(S0, K_only = True)  # Asimov data
        assert n == 1
        assert n0 == 1
        ncomp = SF._ncomp
        free_theta = [i != 0 for i in range(ncomp)]
        theta = [1. if i == 0 else 0. for i in range(ncomp)]
        theta0 = theta  # Does not matter, since we use mu_overwrite
        mu = SF.mu(theta)  # Overwrites *model predictions*
        lnL = SF0.profile_lnL(theta, theta0, epsilon = 1e-3, free_theta = free_theta,
                mu_overwrite = mu)
        return lnL

    def getfield(self, Sfunc, x_values, y_values):
        ix = 0
        iy = 1
        theta0 = [None, None]
        FF = self._ff_factory(Sfunc, theta0)
        tf =FF.TensorField(ix, iy, x_values, y_values, theta0 = theta0)
        return tf

    def getMinuit(self, Sfunc, theta0, **kwargs):
        FF = self._ff_factory(Sfunc, theta0)
        M = FF.iminuit(theta0, **kwargs)
        return M

    def Delta(self, S, S0, use_lnL = False):
        if use_lnL:
            d2 = -2*(self.lnL(S, S0) - self.lnL(S0, S0))
            return d2
        else:
            eS, N = self.equivalentshapes(S)
            eS0, N0 = self.equivalentshapes(S0)
            d2 = ((eS-eS0)**2/(N0+N)*2).sum()
            return d2

    @staticmethod
    def linearize(Sfunc, x, dx = None):
        S0 = Sfunc(*x)
        gradS = _func_to_templates(Sfunc, x, dx)
        return gradS, S0

# TODO:
# - Make sure Fishpy can be instantiated with arbitrary background and signal
# functions (linearized background).
