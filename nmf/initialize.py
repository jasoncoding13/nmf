#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 14:40:12 2019

@author: jason
@E-mail: jasoncoding13@gmail.com
@Github: jasoncoding13
"""

import numpy as np
from math import sqrt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from .utils import squared_norm
ADD_R = 0.25  # Add a value on R to make R denser if it uses kmeans_init like.


def random_init(n_components, X):
    """initialize dictionary and representation by random positive number.
    Args
        n_components: int.
        X: array with shape of [feature size, sample size].
    Rets
        D: initialized array with shape of [feature size, n_components].
        R: initialized array with shape of [n_components, sample size].
    """
    n_features, n_samples = X.shape
    avg = np.sqrt(X.mean() / n_components)
    rng = np.random.RandomState(13)
    D = avg * rng.randn(n_features, n_components)
    R = avg * rng.randn(n_components, n_samples)
    np.abs(D, out=D)
    np.abs(R, out=R)
    return D, R


def kmeans_init(n_components, X):
    """initialize dictionary and representation by Kmeans.
    Args
        n_components: int.
        X: array with shape of [feature size, sample size].
    Rets
        D: initialized array with shape of [feature size, n_components],
            each column of D is the centres returned by kmeans.
        R: initialized array with shape of [n_components, sample size],
            each row is an one hot encoded vector representing the cluster it
            belongs to.
    """
    n_features, n_samples = X.shape
    rs = np.random.RandomState(13)
    kmeans = KMeans(n_clusters=n_components, random_state=rs).fit(X.T)
    D = kmeans.cluster_centers_.T
    R = np.zeros([n_components, n_samples])
    R[np.array(kmeans.labels_), np.arange(n_samples)] = 1
    R += ADD_R
    return D, R


def pca_kmeans_init(n_components, X):
    """initialize dictionary and representation by PCA and Kmeans. It is
        similar yo kmeans_init and perorms a PCA firstly.
    Args
        n_components: int.
        X: array with shape of [feature size, sample size].
    Rets
        D: initialized array with shape of [feature size, n_components],
            each column of D is the centres returned by kmeans.
        R: initialized array with shape of [n_components, sample size],
            each row is an one hot encoded vector representing the cluster it
            belongs to.
    """
    n_features, n_samples = X.shape
    X_pca = PCA().fit_transform(X.T)
    kmeans = KMeans(n_clusters=n_components).fit(X_pca)
    D = np.zeros([n_features, n_components])
    for i in range(n_components):
        D[:, i] = np.mean(X[:, (kmeans.labels_ == i)], axis=1)
    R = np.zeros([n_components, n_samples])
    R[np.array(kmeans.labels_), np.arange(n_samples)] = 1
    R += ADD_R
    return D, R


def nndsvd_init(n_components, X):
    """initialize dictionary and representation by random positive number.
    Args
        n_components: int.
        X: array with shape of [feature size, sample size].
    Rets
        D: initialized array with shape of [feature size, n_components].
        R: initialized array with shape of [n_components, sample size].
    """
    U, s, V = np.linalg.svd(X, full_matrices=False)
    U = U[:, :n_components]
    s = s[:n_components]
    V = V[:n_components, :]
    D, R = np.zeros(U.shape), np.zeros(V.shape)
    # The leading singular triplet is non-negative
    # so it can be used as is for initialization.
    D[:, 0] = np.sqrt(s[0]) * np.abs(U[:, 0])
    R[0, :] = np.sqrt(s[0]) * np.abs(V[0, :])
    for j in range(1, n_components):
        x, y = U[:, j], V[j, :]
        # extract positive and negative parts of column vectors
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))
        # and their norms
        x_p_nrm, y_p_nrm = sqrt(squared_norm(x_p)), sqrt(squared_norm(y_p))
        x_n_nrm, y_n_nrm = sqrt(squared_norm(x_n)), sqrt(squared_norm(y_n))
        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm
        # choose update
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n
        lbd = np.sqrt(s[j] * sigma)
        D[:, j] = lbd * u
        R[j, :] = lbd * v
    D[D < 1e-6] = 0
    R[R < 1e-6] = 0
    return D, R
