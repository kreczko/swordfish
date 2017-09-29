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
- Effective signal and background counts
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

        B = np.array(B, dtype='float64')
        assert B.shape == (n_bins,), "B has incorrect shape."

        if E is None:
            E = np.ones(n_bins, dtype='float64')
        else:
            E = np.array(E, dtype='float64')
            assert E.shape == (n_bins,), "E has incorrect shape."

        K = la.aslinearoperator(K) if K is not None else None
        assert K.shape == (n_bins, n_bins), "K has incorrect shape."

        T = self._get_constraints(T, n_comp)

        self._flux = S
        self._noise = B 
        self._exposure = E
        if K is not None:
            self._sysflag = True
            self._systematics = K
        else:
            self._sysflag = False
            self._systematics = la.LinearOperator((n_bins, n_bins), 
                    matvec = lambda x: np.zeros_like(x))
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
        if self._solver == "direct":
            dense = D(np.eye(self._nbins))
            invD = np.linalg.linalg.inv(dense)
            for i in range(self._ncomp):
                x[i] = np.dot(invD, self._flux[i]*exposure)*exposure
        elif self._solver == "cg":
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
        lnL = (mu0*np.log(mu)-mu-gammaln(mu0+1)).sum()
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
            return self.lnL(theta, theta0, dmu = None, mu_overwrite =
                    mu_overwrite)
        result = fmin_l_bfgs_b(f, x0, fprime, approx_grad = False)
        if self._verbose:
            print "Best-fit parameters:", result[0]
        if self._at_bound:
            print "WARNING: No maximum with non-negative flux found."
            return -result[1]
        else:
            return -result[1]


class EffectiveCounts(object):
    """*Effective counts* analysis based on a `Swordfish` instance.

    The effective counts method can be used to derive
    
    - expected upper limits
    - discovery reach
    - effective signal and background counts
    
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

    def totalcounts(self, i, theta):
        # FIXME: theta as float or list?
        """Return total signal and background counts.

        Parameters
        ----------
        * `i` [integer]:
            Index of component of interest.
        * `theta` [vector-like, shape=(n_comp)]:
            Normalization of component i.

        Returns
        -------
        * `s` [float]:
            Total signal counts.
        * `b` [float]:
            Total background counts.
        """
        s = sum(self._model._flux[i]*self._model._exposure*theta)
        b = sum(self._model._noise*self._model._exposure)
        return s, b

    def effectivecounts(self, i, theta):
        # FIXME: theta as float or list?
        """Return effective signal and background counts.

        Parameters
        ----------
        * `i` [integer]:
            Index of component of interest.
        * `theta` [vector-like, shape=(n_comp)]:
            Normalization of component i.

        Returns
        -------
        * `seff` [float]:
            Effective signal counts.
        * `beff` [float]:
            Effective background counts.
        """
        I0 = 1./self._model.variance(i)
        thetas = np.zeros(self._model._ncomp)
        thetas[i] = theta
        I = 1./self._model.variance(i, theta = thetas)
        if I0 == I:
            return 0., None
        seff = 1/(1/I-1/I0)*theta**2
        beff = 1/I0/(1/I-1/I0)**2*theta**2
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
                    s, b = self.effectivecounts(i, theta = theta)
                    if s == 0: b = 1.
                    z_list.append(s/np.sqrt(s+b))
                    if z_list[-1] > Z:
                        break
                    else:
                        theta_list.append(theta*1.3)
                    #print theta, z_list
                thetaUL = np.interp(Z, z_list, theta_list)
            return thetaUL

    def discoveryreach(self, i, alpha = 2e-6, force_gaussian = False):
        #FIXME: Check definition of alpha
        r"""Return expected discovery reach.

        Parameters
        ----------
        * `i` [integer]:
            Index of component of interest.
        * `alpha` [float]:
            Statistical significance.  For example, 2-e6 means a 5-sigma
            detection.
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
            s, b = self.effectivecounts(i, theta = theta)
            if s == 0: b = 1.
            z_list.append((s+b)*np.log((s+b)/b)-s)
            if z_list[-1] > Z**2/2:
                break
            else:
                theta_list.append(theta*1.3)
        return np.interp(Z**2/2, z_list, theta_list)


class Funkfish(object):
    r"""`Swordfish`, `EffectiveCounts` and `iminuit`-factory, based on non-linear models.

    The underlying likelihood function is identical to the one of `Swordfish`,
    with the only difference that model expectations are derived from
    $$
    \mu_i(\vec\theta, \delta \vec B) = \left[ f_i(\vec\theta) + \delta B_i\right]\cdot E_i
    $$

    `Funkfish` can generate `Swordfish` and `EffectiveCounts` objects as local
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
        flux = self._func_to_templates(self._f, x0)
        noise = self._f(*x0)
        return Swordfish(flux, noise, self._exposure, self._Sigma, T = self._constraints)

    def EffectiveCounts(self, theta0 = None):
        """Generate `EffectiveCounts` instance.

        Directly generates `EffectiveCounts` instance from `Swordfish`
        instance.  See documentation of `get_Swordfish` method .
        """
        SF = self.Swordfish(theta0)
        return EffectiveCounts(SF)

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
        #FIXME: Fix treatment of theta0
        theta0 = theta0.copy()
        g = np.zeros((len(y_bins), len(x_bins), 2, 2))
        for i, y in enumerate(y_bins):
            for j, x in enumerate(x_bins):
                theta0[ix] = x
                theta0[iy] = y
                SF = self.Swordfish(x0_dict)
                g[i, j] = SF.effectivefishermatrix((ix, iy))
        return mp.TensorField(x_bins, y_bins, g)

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
        x0err = np.where(x0>0., x0*0.01, 0.01)
        M = self._init_minuit(chi2, x = x0, x_err = x0err, **kwargs)
        return M
