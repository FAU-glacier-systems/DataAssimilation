# -*- coding: utf-8 -*-
# pylint: disable=invalid-name, too-many-arguments, too-many-instance-attributes
# pylint: disable=attribute-defined-outside-init

"""Copyright 2015 Roger R Labbe Jr.

FilterPy library.
http://github.com/rlabbe/filterpy

Documentation at:
https://filterpy.readthedocs.org

Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

This is licensed under an MIT license. See the readme.MD file
for more information.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import threading
import copy
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from numpy import array, zeros, eye, dot
from numpy.random import multivariate_normal
from filterpy.common import pretty_str, outer_product_sum
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf


class EnsembleKalmanFilter(object):
    """
    This implements the ensemble Kalman filter (EnKF). The EnKF uses
    an ensemble of hundreds to thousands of state vectors that are randomly
    sampled around the estimate, and adds perturbations at each update and
    predict step. It is useful for extremely large systems such as found
    in hydrophysics. As such, this class is admittedly a toy as it is far
    too slow with large N.

    There are many versions of this sort of this filter. This formulation is
    due to Crassidis and Junkins [1]. It works with both linear and nonlinear
    systems.

    Parameters
    ----------

    x : np.array(dim_x)
        state mean

    P : np.array((dim_x, dim_x))
        covariance of the state

    dim_z : int
        Number of of measurement inputs. For example, if the sensor
        provides you with position in (x,y), dim_z would be 2.

    dt : float
        time step in seconds

    N : int
        number of sigma points (ensembles). Must be greater than 1.

    K : np.array
        Kalman gain

    hx : function hx(x)
        Measurement function. May be linear or nonlinear - converts state
        x into a measurement. Return must be an np.array of the same
        dimensionality as the measurement vector.

    fx : function fx(x, dt)
        State transition function. May be linear or nonlinear. Projects
        state x into the next time period. Returns the projected state x.


    Attributes
    ----------
    x : numpy.array(dim_x, 1)
        State estimate

    P : numpy.array(dim_x, dim_x)
        State covariance matrix

    x_prior : numpy.array(dim_x, 1)
        Prior (predicted) state estimate. The *_prior and *_post attributes
        are for convienence; they store the  prior and posterior of the
        current epoch. Read Only.

    P_prior : numpy.array(dim_x, dim_x)
        Prior (predicted) state covariance matrix. Read Only.

    x_post : numpy.array(dim_x, 1)
        Posterior (updated) state estimate. Read Only.

    P_post : numpy.array(dim_x, dim_x)
        Posterior (updated) state covariance matrix. Read Only.

    z : numpy.array
        Last measurement used in update(). Read only.

    R : numpy.array(dim_z, dim_z)
        Measurement noise matrix

    Q : numpy.array(dim_x, dim_x)
        Process noise matrix

    fx : callable (x, dt)
        State transition function

    hx : callable (x)
        Measurement function. Convert state `x` into a measurement

    K : numpy.array(dim_x, dim_z)
        Kalman gain of the update step. Read only.

    inv : function, default numpy.linalg.inv
        If you prefer another inverse function, such as the Moore-Penrose
        pseudo inverse, set it to that instead: kf.inv = np.linalg.pinv

    Examples
    --------

    .. code-block:: Python

        def hx(x):
           return np.array([x[0]])

        F = np.array([[1., 1.],
                      [0., 1.]])
        def fx(x, dt):
            return np.dot(F, x)

        x = np.array([0., 1.])
        P = np.eye(2) * 100.
        dt = 0.1
        f = EnsembleKalmanFilter(x=x, P=P, dim_z=1, dt=dt,
                                 N=8, hx=hx, fx=fx)

        std_noise = 3.
        f.R *= std_noise**2
        f.Q = Q_discrete_white_noise(2, dt, .01)

        while True:
            z = read_sensor()
            f.predict()
            f.update(np.asarray([z]))

    See my book Kalman and Bayesian Filters in Python
    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

    References
    ----------

    - [1] John L Crassidis and John L. Junkins. "Optimal Estimation of
      Dynamic Systems. CRC Press, second edition. 2012. pp, 257-9.
    """

    def __init__(self, x, P, dim_z, dt, N, start_year):
        if dim_z <= 0:
            raise ValueError('dim_z must be greater than zero')

        if N <= 0:
            raise ValueError('N must be greater than zero')

        dim_x = len(x)
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dt = dt
        self.N = N
        #self.hx = hx
        #self.fx = fx
        self.K = zeros((dim_x, dim_z))
        self.z = array([[None] * self.dim_z]).T
        self.S = zeros((dim_z, dim_z))  # system uncertainty
        self.SI = zeros((dim_z, dim_z))  # inverse system uncertainty

        self.initialize(x, P)
        self.Q = eye(dim_z)  # process uncertainty
        self.R = eye(dim_z)  # state uncertainty
        self.inv = np.linalg.inv

        # used to create error terms centered at 0 mean for
        # state and measurement
        self._mean = zeros(dim_x)
        self._mean_z = zeros(dim_z)

        self.year = start_year

    def initialize(self, x, P):
        """
        Initializes the filter with the specified mean and
        covariance. Only need to call this if you are using the filter
        to filter more than one set of data; this is called by __init__

        Parameters
        ----------

        x : np.array(dim_z)
            state mean

        P : np.array((dim_x, dim_x))
            covariance of the state
        """

        if x.ndim != 1:
            raise ValueError('x must be a 1D array')
        print("Initialize ensemble")
        """
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=self.dim_x)
        sample = sampler.random(n=self.N)
        self.sigmas = qmc.scale(sample, x - np.sqrt(np.diag(P)), x + np.sqrt(np.diag(P)))
        """
        self.sigmas = multivariate_normal(mean=x, cov=P, size=self.N)

        """
        self.sigmas = []
        for i in range(self.N):
            sigma = copy.copy(x)
            noise_ela = np.random.normal(0, np.sqrt(P[0, 0]))
            noise_grad_abl = np.random.normal(0, np.sqrt(P[1, 1]))
            noise_grad_acc = np.random.normal(0, np.sqrt(P[2, 2]))
            sigma[0] += noise_ela
            sigma[1] += noise_grad_abl
            sigma[2] += noise_grad_acc
            #sigma = abs(sigma)  # these SMB parameters are always positive
            self.sigmas.append(sigma)
        """

        self.sigmas = np.ma.masked_array(self.sigmas)
        self.x = x
        self.P = P

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def predict(self, ensemble_members):
        """ Predict next position. """

        N = self.N
        dt = self.dt
        year = int(self.year)

        devices = tf.config.list_physical_devices('GPU')

        # if devices:
        if False:
            print("GPU is available.")
            for i, s in enumerate(self.sigmas):
                member = ensemble_members[i]
                self.sigmas[i] = member.forward(s, self.dt)

        else:
            print("No GPU found.")

            def task(member, s, dt):
                member.forward(s, dt)

            # Create a thread pool
            with ThreadPoolExecutor() as executor:
                # Submit tasks to the thread pool
                futures = [executor.submit(task, ensemble_members[i], s, dt) for i, s
                           in enumerate(self.sigmas)]

                # Wait for all tasks to complete
                for future in futures:
                    future.result(timeout=3600)

            # for i, s in enumerate(self.sigmas):
            #    task(s, dt, i, year)

        # forward SMB parameters
        e = multivariate_normal(self._mean, self.Q, N)
        self.sigmas += e
        self.x = np.mean(self.sigmas, axis=0)

        self.P = outer_product_sum(self.sigmas - self.x) / (N - 1)

        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

    def update(self, z, ensemble_members, observation_points, e_r, R=None, inflation=1.0):
        """
        Add a new measurement (z) to the kalman filter. If z is None, nothing
        is changed.

        Parameters
        ----------

        z : np.array
            measurement for this update.

        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise self.R will be used.
        """

        if z is None:
            self.z = array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return

        if R is None:
            R = self.R
        if np.isscalar(R):
            R = eye(self.dim_z) * R

        N = self.N
        dim_z = len(z)
        sigmas_h = zeros((N, dim_z))

        # transform sigma points into measurement space
        for i, member in enumerate(ensemble_members):
            sigmas_h[i] = member.observe(observation_points)

        z_mean = np.mean(sigmas_h, axis=0)

        P_zz = (outer_product_sum(sigmas_h - z_mean) / (N - 1)) + R

        P_xz = outer_product_sum(
            self.sigmas - self.x, sigmas_h - z_mean) / (N - 1)

        self.S = P_zz
        self.SI = self.inv(self.S)
        self.K = dot(P_xz, self.SI)

        P_zz_show = copy.copy(self.K)
        P_zz_show[P_zz_show == 0] = None
        # fig, ax = plt.subplots(figsize=(10, 3))
        # im = ax.imshow(P_zz_show, vmin=-1, vmax=1, cmap='coolwarm')
        # plt.colorbar(im, orientation='horizontal', ax=ax)
        # fig.savefig("Plots/Kalman_Gain" + str(self.year) + ".png")

        #e_r = multivariate_normal(self._mean_z, R, N)
        for i in range(N):

            self.sigmas[i] += dot(self.K, z + e_r[i] - sigmas_h[i])
            #self.sigmas[i] = abs(self.sigmas[i])

        self.x = np.mean(self.sigmas, axis=0)

        ### INFLATION ###
        # Apply multiplicative inflation to deviations from the ensemble mean
        for i in range(N):
            self.sigmas[i] = self.x + inflation * (self.sigmas[i] - self.x)


        self.P = outer_product_sum(self.sigmas - self.x) / (N - 1)
        #self.P = self.P - dot(dot(self.K, self.S), self.K.T)

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()


    def update_etkf(self, observations, ensemble_members, observation_points, e_r, R=None, inflation_factor=1.0):

        # Ensemble size and dimensions
        N = self.N
        xf = self.x
        Xf = np.array(self.sigmas)
        HXf = zeros((self.N, len(observations)))

        # transform sigma points into measurement space
        for i, member in enumerate(ensemble_members):
            HXf[i] = member.observe(observation_points)

        Hxf = np.mean(HXf, axis=0)
        HXf = HXf.T

        d = observations - Hxf
        HX_per_f = HXf - Hxf[:, np.newaxis]
        C = np.linalg.inv(R) @ HX_per_f
        A1 = (N-1)*np.eye(N)
        A2 = A1 + HX_per_f.T @ C

        X_per_f = Xf - xf
        D = C.T @ d
        eigs, ev = np.linalg.eigh(A2)

        # compute perturbations
        Wp1 = np.diag(np.sqrt(1 / eigs)) @ ev.T
        Wp = ev @ Wp1 * np.sqrt(N - 1)

        # differing from pseudocode
        D1 = np.linalg.inv(R) @ d
        D2 = HX_per_f.T @ D1
        wm = ev @ np.diag(1 / eigs) @ ev.T @ D2  # / np.sqrt(Ne-1)

        # Adding perturbed and mean weights
        W = Wp + wm[:, None]

        # final adding up (most costly operation)
        Xa = xf[:, None] + np.array(X_per_f.T) @ W

        self.sigmas = Xa.T
        self.x = xf

        ### INFLATION ###
        # Apply multiplicative inflation to deviations from the ensemble mean
        for i in range(N):
            self.sigmas[i] = self.x + inflation_factor * (self.sigmas[i] - self.x)

        return Xa




    def __repr__(self):
        return '\n'.join([
            'EnsembleKalmanFilter object',
            pretty_str('dim_x', self.dim_x),
            pretty_str('dim_z', self.dim_z),
            pretty_str('dt', self.dt),
            pretty_str('x', self.x),
            pretty_str('P', self.P),
            pretty_str('x_prior', self.x_prior),
            pretty_str('P_prior', self.P_prior),
            pretty_str('Q', self.Q),
            pretty_str('R', self.R),
            pretty_str('K', self.K),
            pretty_str('S', self.S),
            pretty_str('sigmas', self.sigmas),
            pretty_str('hx', self.hx),
            pretty_str('fx', self.fx)
        ])
