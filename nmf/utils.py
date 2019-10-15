#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 10:17:03 2019

@author: jason
@E-mail: jasoncoding13@gmail.com
@Github: jasoncoding13
"""

import numpy as np
import sys
from zipfile import ZipFile


def compute_image_shape(data, reduce=None):
    """Lazily compute the shape of image on different dataset.
    Args
        data: str, 'ORL' or 'EYB' or 'AR'.
        reduce: scale factor.
    Rets
        image_shape: array, [image height, image width].
        reduce
    """
    if data == 'ORL':
        if not reduce:
            reduce = 3
        image_shape = [i // reduce for i in (112, 92)]
    elif data == 'EYB':
        if not reduce:
            reduce = 4
        image_shape = [i // reduce for i in (192, 168)]
    elif data == 'AR':
        if not reduce:
            reduce = 3
        image_shape = [i // reduce for i in (165, 120)]
    return np.array(image_shape), reduce


def print_log(string):
    """Print string into standard output
    Args
        string: str.
    """
    sys.stdout.write(string+'\n')
    sys.stdout.flush()


def squared_norm(x):
    """Squared Euclidean or Frobenius norm of x.
    Args
        x: array_like.
    Rets
    float
        The Euclidean norm when x is a vector, the Frobenius norm when x is a
        matrix (2-d array).
    """
    x = np.ravel(x, order='K')
    return np.dot(x, x)


def unzip_file(source_path, target_path):
    """Unzip zip file
    Args
        source_path: path to be extracted.
        target_path: path to be stored.
    """
    with ZipFile(source_path, 'r') as f:
        f.extractall(path=target_path)
