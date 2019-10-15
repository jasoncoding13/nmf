#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 19:39:06 2019

@author: jason
@E-mail: jasoncoding13@gmail.com
@Github: jasoncoding13
"""

import numpy as np
from .initialize import random_init
from .initialize import nndsvd_init
from .initialize import kmeans_init
from .initialize import pca_kmeans_init
from .utils import print_log
MAX_STEPS = 200
SKIP_STEPS = 10
TOL = 1e-4


class BaseNMF():
    """Base Class
    """
    def __init__(
            self,
            n_components,
            init,
            tol,
            max_iter,
            skip_iter):
        self.n_components = n_components
        self.init = init
        self.tol = tol
        self.max_iter = max_iter
        self.skip_iter = skip_iter

    def _compute_loss(self, X, D, R):
        return None

    def _update(self, X, D, R):
        return None

    def _init(self, X):
        if self.init == 'random':
            D, R = random_init(self.n_components, X)
        elif self.init == 'nndsvd':
            D, R = nndsvd_init(self.n_components, X)
        elif self.init == 'kmeans':
            D, R = kmeans_init(self.n_components, X)
        elif self.init == 'pcakmeans':
            D, R = pca_kmeans_init(self.n_components, X)
        return D, R

    def fit(self, X):
        """
        Args
            X: array with shape of [feature size, sample size].
        Rets
            D: array with shape of [feature size, n_components].
            R: array with shape of [n_components, sample size].
        """
        D, R = self._init(X)
        losses = [self._compute_loss(X, D, R)]
        for iter_ in range(self.max_iter):
            D, R = self._update(X, D, R)
            # check converagence
            if iter_ % self.skip_iter == 0:
                losses.append(self._compute_loss(X, D, R))
                criterion = abs(losses[-1] - losses[-2]) / losses[-2]
                print_log('iter-{:>4}, criterion-{:0<10.5}, {:>13}'.format(
                        iter_, criterion, losses[-1]))
                if criterion < TOL:
                    break
        return D, R


class WeightedNMF(BaseNMF):
    def __init__(
            self,
            n_components,
            init='random',
            tol=1e-4,
            max_iter=200,
            skip_iter=10):
        super().__init__(n_components, init, tol, max_iter, skip_iter)

    def _compute_loss(self, X, D, R):
        return None

    def _update_weight(self, X, D, R):
        return None

    def _update(self, X, D, R):
        # update W
        W = self._update_weight(X, D, R)
        # update D
        denominator_D = (W * D.dot(R)).dot(R.T)
        denominator_D[denominator_D == 0] = np.finfo(np.float32).eps
        D = D * ((W * X).dot(R.T)) / denominator_D
        # update R
        denominator_R = D.T.dot(W * D.dot(R))
        denominator_R[denominator_R == 0] = np.finfo(np.float32).eps
        R = R * (D.T.dot(W * X)) / denominator_R
        return D, R


class NMF(WeightedNMF):
    """ Standard NMF
    """
    def _compute_loss(self, X, D, R):
        return np.linalg.norm(X - D.dot(R)) ** 2

    def _update_weight(self, X, D, R):
        return 1


class CIM_NMF(WeightedNMF):
    """ CIM-NMF
    """
    def _compute_loss(self, X, D, R):
        twice_sigma_square = np.mean(np.square(X - D.dot(R)))
        return np.sum(1 - 1 / np.sqrt(np.pi * twice_sigma_square) *
                      (np.exp(-np.square(X - D.dot(R)) / twice_sigma_square)))

    def _update_weight(self, X, D, R):
        E_square = np.square(X - D.dot(R))
        twice_sigma_square = np.mean(E_square)
        return np.exp(- E_square / twice_sigma_square)


class HuberNMF(WeightedNMF):
    """Huber-NMF
    """
    def _compute_loss(self, X, D, R):
        abs_E = np.abs(X - D.dot(R))
        abs_E[abs_E == 0] = np.finfo(np.float32).eps
        c = np.median(abs_E)
        return np.sum(np.where(abs_E < c,
                               np.square(abs_E),
                               2 * c * abs_E - (c ** 2)))

    def _update_weight(self, X, D, R):
        abs_E = np.abs(X - D.dot(R))
        abs_E[abs_E == 0] = np.finfo(np.float32).eps
        c = np.median(abs_E)
        return np.where(abs_E < c, 1, c / abs_E)


class L1NMF(WeightedNMF):
    """L1-NMF
    """
    def _compute_loss(self, X, D, R):
        return np.sum(np.abs(X - D.dot(R)))

    def _update_weight(self, X, D, R):
        # `eps` cannot be a too small value like np.finfo(np.float32).eps
        eps = X.var() / D.shape[1]
        return 1 / (np.sqrt(np.square(X - D.dot(R))) + eps ** 2)


class L21NMF(WeightedNMF):
    """L21-NMF
    """
    def _compute_loss(self, X, D, R):
        return np.sum(np.sqrt(np.sum(np.square(X - D.dot(R)), axis=0)))

    def _update_weight(self, X, D, R):
        return 1 / np.sqrt(np.sum(np.square(X - D.dot(R)), axis=0))


class RNMF_L1(BaseNMF):
    """RNMF-L1
    Args:
        labmbda_: float, regularization parameter for L1 term. If it is 128
            for images in range from 0 to 255, that is equal to the case when
            lambda_ = 0.5 with images in range from 0 to 1.
    """
    def __init__(
            self,
            n_components,
            init='random',
            tol=1e-4,
            max_iter=200,
            skip_iter=10,
            lambda_=0):
        self.lambda_ = lambda_
        super().__init__(n_components, init, tol, max_iter, skip_iter)

    def _compute_loss(self, X, D, R, E):
        return (np.linalg.norm(X - D.dot(R) - E) ** 2 +
                self.lambda_ * np.sum(np.abs(E)))

    def _update(self, X, D, R):
        # compute E
        E = X - D.dot(R)
        index_greater = E > self.lambda_ / 2
        E[index_greater] = E[index_greater] - self.lambda_ / 2
        index_less = E < -self.lambda_ / 2
        E[index_less] = E[index_less] + self.lambda_ / 2
        E[np.logical_not(np.logical_or(index_greater, index_less))] = 0
        # update D
        E_minus_X = E - X
        E_minus_X_dot_RT = (E_minus_X).dot(R.T)
        denominator_D = 2 * D.dot(R).dot(R.T)
        denominator_D[denominator_D == 0] = np.finfo(np.float32).eps
        D = (D * (np.abs(E_minus_X_dot_RT) - E_minus_X_dot_RT) /
             denominator_D)
        # update R
        DT_dot_E_minus_X = D.T.dot(E_minus_X)
        denominator_R = 2 * D.T.dot(D).dot(R)
        denominator_R[denominator_R == 0] = np.finfo(np.float32).eps
        R = (R * (np.abs(DT_dot_E_minus_X) - DT_dot_E_minus_X) /
             denominator_R)
        # normalize D and R
        # keepdims=True makes its shape to be [1, n_components]
        normalization = np.sqrt(np.sum(np.square(D), axis=0, keepdims=True))
        D = D / normalization
        R = R * normalization.T
        return D, R, E

    def fit(self, X):
        """
        Args
            X: array with shape of [feature size, sample size].
        Rets
            D: array with shape of [feature size, n_components].
            R: array with shape of [n_components, sample size].
            E: array, a error matrix to capture the sprase corruption.
        """
        D, R = self._init(X)
        losses = [self._compute_loss(X, D, R, X - D.dot(R))]
        for iter_ in range(self.max_iter):
            D, R, E = self._update(X, D, R)
            # check converagence
            if iter_ % self.skip_iter == 0:
                losses.append(self._compute_loss(X, D, R, E))
                criterion = abs(losses[-1] - losses[-2]) / losses[-2]
                print_log('iter-{:>4}, criterion-{:0<10.5}, {:>13}'.format(
                        iter_, criterion, losses[-1]))
                if criterion < TOL:
                    break
        return D, R, E


class RCNMF(BaseNMF):
    """RCNMF
    Args:
        theta: int or float, the thresholding parameter for choosing the
            extreme data outliers. It cannot be too small otherwise it treats
            many samples as outliers. It default value is `float(inf)` which
            means there are no outliers.
    """
    def __init__(
            self,
            n_components,
            init='random',
            tol=1e-4,
            max_iter=200,
            skip_iter=10,
            theta=float('inf')):
        self.theta = theta
        super().__init__(n_components, init, tol, max_iter, skip_iter)

    def _compute_loss(self, X, D, R):
        return np.sum(
                np.minimum(np.sqrt(np.sum(np.square(X - D.dot(R)), axis=0)),
                           self.theta))

    def _update(self, X, D, R):
        # compute diagonal matrix W
        # W is not initialized as the identity matrix as paper proposes.
        col_norm_E = np.sqrt(np.sum(np.square(X - D.dot(R)), axis=0))
        col_norm_E[col_norm_E == 0] = np.finfo(np.float32).eps
        W = np.diag(np.where(col_norm_E <= self.theta, 1/(2*col_norm_E), 0))
        # update D
        denominator_D = (D.dot(R).dot(W).dot(R.T))
        denominator_D[denominator_D == 0] = np.finfo(np.float32).eps
        D = D * (X.dot(W).dot(R.T)) / denominator_D
        # update R
        sqrt_denominator_R = (D.T.dot(X).dot(R.T).dot(R).dot(W))
        sqrt_denominator_R[sqrt_denominator_R == 0] = np.finfo(np.float32).eps
        R = R * np.sqrt((D.T.dot(X).dot(W)) / sqrt_denominator_R)
        return D, R


class HCNMF(BaseNMF):
    """HCNMF
    Args:
        alpha: float, initial learning rate to update D.
        beta: float, initial learning rate to update R.
    """
    def __init__(
            self,
            n_components,
            init='random',
            tol=1e-4,
            max_iter=200,
            skip_iter=10,
            alpha=0.001,
            beta=0.001):
        self.alpha = alpha
        self.beta = beta
        super().__init__(n_components, init, tol, max_iter, skip_iter)

    def _compute_loss(self, X, D, R):
        return np.sum(np.sqrt(1 + np.square(X - D.dot(R))) - 1)

    def _update(self, X, D, R):
        denominator = np.sqrt(1+np.linalg.norm(X - D.dot(R)))
        # update D by Armijo rule
        grad_D = (D.dot(R).dot(R.T) - X.dot(R.T)) / denominator
        D_updated = D - self.alpha * grad_D
        while self._compute_loss(X, D_updated, R) > self._compute_loss(X, D, R):
            self.alpha *= 0.5
            D_updated = D - self.alpha * grad_D
        D = D_updated
        # update D by Armijo rule
        grad_R = (D.T.dot(D).dot(R) - D.T.dot(X)) / denominator
        R_updated = R - self.beta * grad_R
        while self._compute_loss(X, D, R_updated) > self._compute_loss(X, D, R):
            self.beta *= 0.5
            R_updated = R - self.beta * grad_R
        R = R_updated
        return D, R
