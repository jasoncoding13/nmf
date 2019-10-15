#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:10:49 2019

@author: jason
@E-mail: jasoncoding13@gmail.com
@Github: jasoncoding13
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import StratifiedKFold
from .preprocess import load_data, add_block_noise, add_salt_noise
from .utils import compute_image_shape, print_log
plt.rc('font', family='serif', size=26)
plt.rc('lines', markerfacecolor='none')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')


def compute_RRE(X, X_hat):
    """Compute relative reconstruction error.
    Args:
        X: array with shape of [feature size, sample size]
        X_hat: array with the same shape as X for reconstruction.
    """
    return np.linalg.norm(X - X_hat) / np.linalg.norm(X)


def compute_ACC_NMI(R, Y):
    """Compute accuracy and normalized mutual information.
    Args:
        R: array with shape of [number of components, sample size], the
            subspace learned from NMF.
        Y: array with shape of [sample size], the true label of raw data.
    """
    kmeans = KMeans(n_clusters=len(set(Y))).fit(R.T)
    Y_pred = np.zeros(Y.shape)
    for i in set(kmeans.labels_):
        ind = (kmeans.labels_ == i)
        # ([label, frequence])[0][0]
        Y_pred[ind] = Counter(Y[ind]).most_common(1)[0][0]
    acc = accuracy_score(Y, Y_pred)
    nmi = normalized_mutual_info_score(Y, Y_pred, average_method='arithmetic')
    return acc, nmi


def compute_metrics(X, X_hat, R, Y):
    """Compute RRE, ACC and NMI.
    """
    acc, nmi = compute_ACC_NMI(R, Y)
    metrics = {'RRE': compute_RRE(X, X_hat),
               'ACC': acc,
               'NMI': nmi}
    print_log('RRE: {RRE}, ACC: {ACC}, NMI:{NMI}'.format(**metrics))
    return metrics


def inspect_dictionary(D, data, reduce=None, n_cols=5):
    """Inspect the dictionary
    Args:
        D: array with shape of [feature size, number of components]
        data: str, 'ORL' or 'EYB' or 'AR'.
        reduce: scale factor.
        n_cols: int, number of images shown in each row.
    """
    image_shape, reduce = compute_image_shape(data, reduce)
    nrows = D.shape[1] // n_cols
    nrows += 1 if D.shape[1] % n_cols else 0
    for i in range(nrows):
        plt.figure(figsize=(16, 9))
        for j in range(n_cols):
            plt.subplot(1, n_cols, j+1)
            plt.imshow(D[:, i*n_cols+j].reshape(image_shape), cmap=plt.cm.gray)
            plt.axis('off')
        plt.show()


def reconstruct(X, X_noised, X_hat, data, reduce=None, ind=None, path=None):
    """Reconstruct the image
    Args:
        X: array with shape of [feature size, sample size], original images.
        X_noised: array like X, noised images.
        X_hat: array like X, reconstructed images.
        data: str, 'ORL' or 'EYB' or 'AR'.
        reduce: scale factor.
        ind: int, index of images to plot.
        path: path to save plot.
    """
    image_shape, reduce = compute_image_shape(data, reduce)
    if not ind:
        if data in ('ORL', 'EYB'):
            ind = np.random.randint(X.shape[1])
        elif data == 'AR':
            ind = np.random.randint(7, 13) + np.random.randint(200) * 13
    plt.figure(figsize=(16, 9))
    plt.subplot(131)
    plt.imshow(X[:, ind].reshape(image_shape), cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('Image(Original)')
    plt.subplot(132)
    plt.imshow(X_noised[:, ind].reshape(image_shape), cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('Image(Noised)')
    plt.subplot(133)
    plt.imshow(X_hat[:, ind].reshape(image_shape), cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('Image(Reconstructed)')
    if path:
        plt.savefig(path)
        print_log(f'Save image at {path}')
    plt.show()


def experiment(model, data, noise, noise_param_lst, reduce=None, n_splits=5):
    """CV experiment
    Args:
        model: instance like `NMF(n_components=40)`.
        data: str, 'ORL' or 'EYB' or 'AR'.
        noise: 'block' or 'salt', type of noise to add.
        noise_param_lst: a list of parameters like [10, 12]
        reduce: scale factor.
        n_splits: int, number of folds for cross validation.
    """
    X, Y = load_data(data=data, reduce=reduce)
    if noise == 'block':
        add_noise_fun = add_block_noise
    elif noise == 'sale':
        add_noise_fun = add_salt_noise
    else:
        raise ValueError("noise should be 'block' or 'salt'")
    _array = np.zeros([len(noise_param_lst), 7+n_splits*3])
    skf = StratifiedKFold(
            n_splits=n_splits, random_state=np.random.RandomState(13))
    module_path = os.path.dirname(__file__)
    for i, noise_param in enumerate(noise_param_lst):
        _row = [noise_param]
        X_noised = add_noise_fun(X, noise_param, data=data, reduce=reduce)
        _row_RRE = []
        _row_ACC = []
        _row_NMI = []
        for j, (train_index, _) in enumerate(skf.split(Y, Y)):
            D, R = model.fit(X_noised[:, train_index])
            X_hat = D.dot(R)
            reconstruct(
                    X[:, train_index],
                    X_noised[:, train_index],
                    X_hat,
                    data=data,
                    reduce=reduce,
                    path='{mp}/plots/{m}_{d}_{n}_{p}_{j}.png'.format(
                            mp=module_path,
                            m=model.__class__.__name__,
                            d=data,
                            n=noise,
                            p=noise_param,
                            j=j))
            dct_metrics = compute_metrics(
                    X[:, train_index], X_hat, R, Y[train_index])
            _row_RRE.append(dct_metrics['RRE'])
            _row_ACC.append(dct_metrics['ACC'])
            _row_NMI.append(dct_metrics['NMI'])
        _row += [np.mean(_row_RRE), np.std(_row_RRE),
                 np.mean(_row_ACC), np.std(_row_ACC),
                 np.mean(_row_NMI), np.std(_row_NMI)]
        _row += _row_RRE
        _row += _row_ACC
        _row += _row_NMI
        _array[i, :] = _row
    df_cv = pd.DataFrame(_array)
    df_cv.columns = (
            ['noise_param',
             'mean_RRE', 'std_RRE',
             'mean_ACC', 'std_ACC',
             'mean_NMI', 'std_NMI'] +
            [f'{i}_{m}'
             for i in range(n_splits)
             for m in ['RRE', 'ACC', 'NMI']])
    csv_path = '{mp}/results/{m}_{d}_{n}.csv'.format(
                            mp=module_path,
                            m=model.__class__.__name__,
                            d=data,
                            n=noise)
    df_cv.to_csv(path_or_buf=csv_path, index=False)
    print_log(f'Save cross validation result at {csv_path}')
    return df_cv


def plot_result(models, data, noise, metric, path=None):
    """Plot the CV result in line charts
    Args:
        models: list of str, names of class.
        data: str, 'ORL' or 'EYB' or 'AR'.
        noise: str, 'block' or 'salt', type of noise to add.
        metric: str, 'RRE' or 'ACC' or 'NMI'.
    """
    fig, ax = plt.subplots(figsize=(16, 9))
    module_path = os.path.dirname(__file__)
    for name in os.listdir(f'{module_path}/results'):
        lst_name = name.split('.')[0].split('_')
        model = '_'.join(lst_name[0:-2])
        if data in name and noise in name and model in models:
            df_cv = pd.read_csv(f'{module_path}/results/{name}')
            ax.plot(df_cv[f'mean_{metric}'], label=model)
    xticklabels = df_cv['noise_param']
    ax.grid(True)
    ax.legend()
    ax.set_title(f'{metric} VS {noise} parameter of different NMFs')
    ax.set_xlabel(f'{noise} parameter')
    ax.set_xticks(range(len(xticklabels)))
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel(metric)
    ax.set_yticklabels(['{:,.2%}'.format(x) for x in ax.get_yticks()])
    plt.show()
