#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 15:24:38 2019

@author: jason
@E-mail: jasoncoding13@gmail.com
@Github: jasoncoding13
"""

import numpy as np
import os
from PIL import Image
from .utils import compute_image_shape
from .utils import print_log
from .utils import unzip_file


def load_data(data, reduce=None):
    """Load dataset to numpy array.
    Args:
        data: str, 'ORL' or 'EYB' or 'AR'.
        reduce: scale factor for zooming out images, shape of image returned
            is [original height // reduce, original width]
    Ret:
        images: array with shape of [feature size, sample size], where feature
            size = height * width.
        labels: array with shape of [sample size].
    """
    # find data
    path = os.path.dirname(__file__)
    if data == 'ORL':
        path += '/data/ORL'
    elif data == 'EYB':
        path += '/data/CroppedYaleB'
    elif data == 'AR':
        path += '/data/AR'
    else:
        raise ValueError("data shoulbe 'ORL' or 'EYB' or 'AR'")
    if os.path.exists(path):
        pass
    else:
        print_log(f'{path} does not exist.')
        path_zip = path+'.zip'
        if os.path.exists(path_zip):
            unzip_file(path_zip, path)
            print_log(f'Unzipped {path_zip} to {path}.')
        else:
            print_log(f'{path_zip} does not exist.')
    image_shape, reduce = compute_image_shape(data, reduce)
    images, labels = [], []
    if data == 'ORL' or data == 'EYB':
        for i, person in enumerate(sorted(os.listdir(path))):
            if not os.path.isdir(os.path.join(path, person)):
                continue
            for fname in os.listdir(os.path.join(path, person)):
                # Remove background images in Extended YaleB dataset.
                if fname.endswith('Ambient.pgm'):
                    continue
                if not fname.endswith('.pgm'):
                    continue
                # load image.
                img = Image.open(os.path.join(path, person, fname))
                img = img.convert('L')  # grey image
                # reduce computation complexity.
                img = img.resize(np.flip(image_shape))
                # convert image to numpy array.
                img = np.asarray(img).reshape((-1, 1))
                # collect data and label.
                images.append(img)
                labels.append(i)
    elif data == 'AR':
        for fname in os.listdir(path):
            if not fname.endswith('.pgm'):
                continue
            # get label.
            label = int(fname[2:5])
            if fname[0] == 'W':  # start from 50
                label += 50
            # load image.
            img = Image.open(os.path.join(path, fname))
            img = img.convert('L')  # grey
            # reduce computation complexity.
            img = img.resize([s//reduce for s in img.size])
            # convert image to numpy array.
            img = np.asarray(img).reshape((-1, 1))
            # collect data and label.
            images.append(img)
            labels.append(label)
    # concate all images and labels.
    images = np.concatenate(images, axis=1)
    labels = np.array(labels)
    print_log('data: {}, images: {}, labels: {}, each image: {}'.format(
            data, images.shape, labels.shape, image_shape))
    return images, labels


def add_block_noise(X, b, data, reduce=None):
    """Add block noise on X.
    Args
        X: array with shape of [feature size, sample size].
        b: int, block size.
        data: str, 'ORL' or 'EYB' or 'AR'.
        reduce: scale factor.
    Rets
        X: array with shape of [feature size, sample size].
    """
    n_feature, n_sample = X.shape
    image_shape, reduce = compute_image_shape(data, reduce)
    X = X.copy().reshape((*image_shape, n_sample))
    # compute the maximum of index where the block can start
    max_block_index_h, max_block_index_w = image_shape - b + 1
    # example:
    # image.shape = (192, 168) b = 10
    # In the dimension of width, block can start in the range of [0, 158].
    # [0, 1, ..., 9, ..., 158, ..., 167]
    #  |  block   |        |  block  |
    # `1` is added because `np.random.randint` excludes the high bound.
    # here it should be `np.random.randint(159)`
    for i in range(n_sample):
        block_index_h = np.random.randint(max_block_index_h)
        block_index_w = np.random.randint(max_block_index_w)
        X[block_index_h:block_index_h+b,
          block_index_w:block_index_w+b,
          i] = 255
    X = X.reshape((n_feature, n_sample))
    return X


def add_salt_noise(X, p, data, reduce=None):
    """Add salt and pepper noise on X.
    Args
        X: array with shape of [feature size, sample size].
        p: float, percentage of pixels are noised.
        data: str, 'ORL' or 'EYB' or 'AR'.
        reduce: scale factor.
    Rets
        X: array with shape of [feature size, sample size].
    """
    n_feature, n_sample = X.shape
    X = X.copy()
    noised_size = int(n_feature * p)
    for i in range(n_sample):
        index = np.arange(n_feature)
        np.random.shuffle(index)
        X[index[:noised_size//2+1], i] = 0
        X[index[noised_size//2+1:noised_size], i] = 255
    return X
