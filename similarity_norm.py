#!/usr/bin/env python      
# -*- coding: utf-8 -*-     
# Author: feng
# refer: https://stackoverflow.com/questions/17969532/how-to-normalize-similarity-measures-from-wordnet
#        https://wiki.apache.org/lucene-java/ScoresAsPercentages Disadvantage of this method!
import numpy as np


# Solution:
# 1, Reference _predict_proba_lr function
def softmax(X):
    prob = X
    prob *= -1
    np.exp(prob, prob)
    prob += 1
    np.reciprocal(prob, prob)
    return prob


# examples:
X = np.array([3.89, 3.13, 2.22, -1.2, 20.3, -3.33])
softmax(X)


# 2, Just use the softmax value as probability
def softmax1(X, v, copy=True):
    '''
    :param X: dot product score of one query with all documents
    :param v: dot product score of one query with itself
    :param copy:
    :return:
    '''
    if copy:
        X = np.copy(X)
    X[X < 0] = 0  # Filter the negative value to keep the output in [0, 1]
    max_prob = v
    X -= max_prob
    np.exp(X, X)
    sum_prob = np.sum(X)
    X /= sum_prob
    return X


def softmax2(X, v, copy=True):
    '''
    :param X: dot product score of one query with all documents
    :param v: dot product score of one query with itself
    :param copy:
    :return:
    '''
    if copy:
        X = np.copy(X)
    X[X < 0] = 0  # Filter the negative value to keep the output in [0, 1]
    max_prob = v
    X -= max_prob
    return np.exp(X, X)

# counter examples:
X = np.array([20.89, 22.13, 23.22, -15.2, 24.3, -20.33])
softmax1(X, 24.3)
# array([0.0222, 0.0768, 0.2284, 0.    , 0.6726, 0.    ])
softmax2(X, 24.3)
# array([0.033 , 0.1142, 0.3396, 0.    , 1.    , 0.    ])

# 3, Just use the similarity value [-1, 1]. Discard the negative part, and use it as probability.





