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
import copy
from scipy.special import gammaln
from scipy.optimize import fmin_l_bfgs_b
import metricplot as mp


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

    def _summedNoise(self, thetas = None):
        noise_tot = self._noise*1.  # Make copy
        if thetas is not None: 
            for i in range(max(self._ncomp, len(thetas))):
                noise_tot += thetas[i]*self._flux[i]
        return noise_tot

    def get_mu(self, thetas = None):
        r"""Return expectation values for given model parameters.

        Parameters
        ----------
        * `thetas` [vector-like, shape=(n_comp), or `None`]:
            Model parameters.  The value `None` implies $\theta_i = 0$.

        Returns
        -------
        * `mu` [vector-like, shape=(n_bins)]:
            Expectation value, $\mu_i(\vec \theta, \delta \vec B = 0)$.
        """
        return self._summedNoise(thetas)*self._exposure

    def _solveD(self, thetas = None):
        """
        Calculates:
            N = noise + thetas*flux
            D = diag(E)*Sigma*diag(E)+diag(N*E)
            x[i] = D^-1 flux[i]*E

        Note: if Sigma = None: x[i] = flux[i]/noise

        Returns:
            x, noise, exposure
        """
        noise = self._summedNoise(thetas)
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

    def fishermatrix(self, thetas = None):
        r"""Return Fisher information matrix.

        Parameters
        ----------
        * `thetas` [vector-like, shape=(n_comp)]:
            Model parameters.  The value `None` is equivalent to
            $\vec\theta=0$.

        Returns
        -------
        * `I` [matrix-like, shape=(n_comp, n_comp)]:
            Fisher information matrix, $\mathcal{I}\_{ij}$.
        """
        x, noise, exposure = self._solveD(thetas=thetas)
        I = np.zeros((self._ncomp,self._ncomp))
        for i in range(self._ncomp):
            for j in range(i+1):
                tmp = sum(self._flux[i]*x[j])
                I[i,j] = tmp
                I[j,i] = tmp
        return I+np.diag(1./self._constraints**2)

    def infoflux(self, thetas = None):
        r"""Return Fisher information flux.

        Parameters
        ----------
        * `thetas` [vector-like, shape=(n_comp)]:
            Model parameters.  The value `None` is equivalent to
            $\vec\theta=0$.

        Returns
        -------
        * `F` [3-D array, shape=(n_comp, n_comp, n_bins)]:
            Fisher information flux.
        """
        x, noise, exposure = self._solveD(thetas=thetas)

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

        Arguments
        ---------
        i : integer
            index of component of interest
        **kwargs
            Passed on to fishermatrix
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
    def lnL(self, thetas, thetas0, dmu = None, epsilon = 1e-3, derivative =
            False, mu_overwrite = None):
        """Return likelihood function, assuming Asimov data.

        Arguments
        ---------
        thetas : array-like, optional
            Definition of flux (added to noise during evaluation).
        thetas0 : array-like, optional
            Definition of Asimov data.
        dmu : array-like
            Systematic deviation.
        epsilon : Float
            Fraction of diagonal noise added to Sigma for stable matrix
            inversion.  Default is 1e-2.
        derivative: bool
            Return also partial derivative w.r.t. thetas and w.r.t. dmu
        """
        thetas = np.array(thetas, dtype='float64')
        thetas0 = np.array(thetas0, dtype='float64')
        mu0 = self._summedNoise(thetas0)*self._exposure
        systnoise = self._summedNoise(thetas0)*epsilon/self._exposure
        if mu_overwrite is None:
            mu =  self._summedNoise(thetas)*self._exposure
        else:
            mu = mu_overwrite.copy()
        if dmu is None:
            dmu = np.zeros_like(self._exposure)
        if self._sysflag:
            mu += dmu*self._exposure
        self._at_bound = any(mu<mu0*1e-6)
        mu = np.where(mu<mu0*1e-6, mu0*1e-6, mu)
        lnL = (mu0*np.log(mu)-mu-gammaln(mu0+1)).sum()
        lnL -= (0.5*thetas**2/self._constraints**2).sum()
        if self._sysflag:
            dense = self._systematics(np.eye(self._nbins))
            #invS = np.linalg.linalg.inv(dense+np.eye(self._nbins)*epsilon)
            invS = np.linalg.linalg.inv(dense+np.diag(systnoise))
            lnL -= 0.5*(invS.dot(dmu)*dmu).sum()
        if derivative:
            dlnL_dtheta = (mu0/mu*self._flux*self._exposure-self._flux*self._exposure).sum(axis=1)
            dlnL_dtheta -= thetas/self._constraints**2
            if self._sysflag:
                dlnL_dmu = mu0/mu*self._exposure - self._exposure - invS.dot(dmu)
            else:
                dlnL_dmu = None
            return lnL, dlnL_dtheta, dlnL_dmu
        else:
            return lnL

    def profile_lnL(self, thetas, thetas0, epsilon = 1e-3, free_thetas = None,
            mu_overwrite = None):
        """Return profile likelihood.

        Arguments
        ---------
        free_thetas : array-like, boolean
            Parameters kept fixed during maximization
        """
        thetas = np.array(thetas, dtype='float64')
        thetas0 = np.array(thetas0, dtype='float64')
        if free_thetas is None:
            free_thetas = np.zeros(len(thetas), dtype='bool')
        else:
            free_thetas = np.array(free_thetas, dtype='bool')
        Nfree_thetas = (free_thetas).sum()
        Nsyst = len(self._exposure) if self._sysflag else 0
        N = Nfree_thetas + Nsyst
        x0 = np.zeros(N)
        x0[:Nfree_thetas] = thetas[free_thetas]
        fp = 0.
        def f(x):
            global fp
            thetas[free_thetas] = x[:Nfree_thetas]
            dmu = x[Nfree_thetas:]
            lnL, grad_theta, grad_dmu = self.lnL(thetas, thetas0, dmu = dmu,
                    epsilon = epsilon, derivative = True, mu_overwrite = mu_overwrite)
            fp = np.zeros(N)
            fp[:Nfree_thetas] = grad_theta[free_thetas]
            fp[Nfree_thetas:] = grad_dmu
            return -lnL
        def fprime(x):
            global fp
            return -fp
        if N == 0.:
            return self.lnL(thetas, thetas0, dmu = None, mu_overwrite =
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
    """EffectiveCounts(model).
    """
    def __init__(self, model):
        """Construct EffectiveCounts object.

        Paramters
        ---------
        model : Swordfish
            Input Swordfish model.

        Note: The functionality applies *only* to additive component models.
        You have been warned.
        """
        self._model = model

    def noise_counts(self):
        """Return total counts.

        Parameters
        ----------
        i : integer
            Component of interest.
        theta : float
            Normalization of component i.

        Returns
        -------
        lambda : float
            Number of counts in component i.
        """
        return sum(self._model._noise*self._model._exposure)

    def counts(self, i, theta):
        """Return total counts.

        Parameters
        ----------
        i : integer
            Component of interest.
        theta : float
            Normalization of component i.

        Returns
        -------
        lambda : float
            Number of counts in component i.
        """
        return sum(self._model._flux[i]*self._model._exposure*theta)

    def effectivecounts(self, i, theta):
        """Return effective counts.

        Parameters
        ----------
        i : integer
            Component of interest.
        theta : float
            Normalization of component i.

        Returns
        -------
        s : float
            Effective signal counts.
        b : float
            Effective backgroundc counts.
        """
        I0 = 1./self._model.variance(i)
        thetas = np.zeros(self._model._ncomp)
        thetas[i] = theta
        I = 1./self._model.variance(i, thetas = thetas)
        if I0 == I:
            return 0., None
        s = 1/(1/I-1/I0)*theta**2
        b = 1/I0/(1/I-1/I0)**2*theta**2
        return s, b

    def upperlimit(self, alpha, i, gaussian = False):
        """Returns upper limits, based on effective counts method.

        Parameters
        ----------
        alpha : float
            Statistical significance (e.g., 95% CL is 0.05).
        i : integer
            Component of interest.
        gaussian : bool, optional
            Force gaussian errors.

        Returns
        -------
        thetaUL : float
            Predicted upper limit on component i.
        """
        Z = stats.norm.isf(alpha)
        I0 = 1./self._model.variance(i)
        if gaussian:
            return Z/np.sqrt(I0)
        else:
            thetas = np.zeros(self._model._ncomp)
            thetaUL_est = Z/np.sqrt(I0)  # Gaussian estimate
            thetas[i] = thetaUL_est
            I = 1./self._model.variance(i, thetas = thetas)
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

    def discoveryreach(self, alpha, i, gaussian = False):
        Z = stats.norm.isf(alpha)
        var0 = self._model.variance(i)
        if gaussian:  # Gaussian approximation
            return Z*var0**0.5

        thetas = np.zeros(self._model._ncomp)
        thetaDT_est = Z*np.sqrt(var0)  # Gaussian approx. as starting point
        thetas[i] = thetaDT_est
        var = self._model.variance(i, thetas = thetas)
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


class FunkFish(object):
    """Docstring for FunkFish"""
    def __init__(self, f, Sigma, exposure, x0, constraints = None):
        self._f = f
        self._Sigma = Sigma
        self._exposure = exposure
        self._x0 = np.array(x0, dtype='float64')
        self._constraints = constraints

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

    def get_Swordfish(self, x = None):
        x0 = self._get_x0(x)
        flux = self._func_to_templates(self._f, x0)
        noise = self._f(*x0)
        return Swordfish(flux, noise, self._exposure, self._Sigma, T = self._constraints)

    def get_EffectiveCounts(self, x0 = None):
        SF = self.get_Swordfish(x0)
        return EffectiveCounts(SF)

    def get_TensorFields(self, i_x, i_y, x_bins, y_bins, x0_dict = {}):
        x0_dict = x0_dict.copy()
        g = np.zeros((len(y_bins), len(x_bins), 2, 2))
        for i, y in enumerate(y_bins):
            for j, x in enumerate(x_bins):
                x0_dict[i_x] = x
                x0_dict[i_y] = y
                SF = self.get_Swordfish(x0_dict)
                g[i, j] = SF.effectivefishermatrix((i_x, i_y))
        return mp.TensorField(x_bins, y_bins, g)

    def get_iminuit(self, x0 = None, **kwargs):
        x0 = self._get_x0(x0)  # make list
        SF0 = self.get_Swordfish(x0)
        def chi2(x):
            mu = self.get_Swordfish(x).get_mu()  # Get proper model prediction
            lnL = SF0.profile_lnL(x-x0, x0*0., mu_overwrite = mu)
            return -2*lnL
        x0err = np.where(x0>0., x0*0.01, 0.01)
        M = self._init_minuit(chi2, x = x0, x_err = x0err, **kwargs)
        return M


