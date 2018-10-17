#!/usr/bin/env python      
# -*- coding: utf-8 -*-     
# Author: feng
# refer: https://stackoverflow.com/questions/17969532/how-to-normalize-similarity-measures-from-wordnet
#        https://wiki.apache.org/lucene-java/ScoresAsPercentages Disadvantage of this method!
import numpy as np


# Solution:
# 4, Supervised method
# 计算两个句子的similarity，本质上是一个无监督方法，且similarity无法归一化为相对概率（如点积）。
# 采用监督学习方法，训练一个线性二分类模型，特征为similarity，y值为0/1，即可获得similarity的概率表示。
# 特征既可以是简单的similarity，也可以是CNN/RNN提取的similarity.

# Representation:
# similarity * coef + intercept = label

# 3, Logistic function - refer _predict_proba_lr
def logistic_norm(X, copy=True):
    '''
    1/(exp(-x)+1)
    :param X:
    :param copy:
    :return:
    '''
    if copy:
        X = np.copy(X)
    prob = X
    prob *= -1
    np.exp(prob, prob)
    prob += 1
    np.reciprocal(prob, prob)
    return prob
# 特点：
# 1）- 若X不小于3，则prob不小于0.95. 是否适用于点击/余弦相似度？


# examples:
X = np.array([3.89, 3.13, 2.22, -1.2, 20.3, -3.33])
logistic_norm(X).round(4)
# array([0.98  , 0.9581, 0.902 , 0.2315, 1.    , 0.0346])
np.reciprocal(np.exp(-(3))+1)
# 0.9525741268224334
np.reciprocal(np.exp(-(-3))+1)
# 0.04742587317756678


# 2, Softmax function - refer softmax.md
def softmax_norm(X, v, copy=True):
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


def softmax(X, v, copy=True):
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
    return X

# counter examples:
X = np.array([20.89, 22.13, 23.22, -15.2, 24.3, -20.33])
softmax_norm(X, 24.3).round(4)
# array([0.0222, 0.0768, 0.2284, 0.    , 0.6726, 0.    ])
softmax(X, 24.3).round(4)
# array([0.033 , 0.1142, 0.3396, 0.    , 1.    , 0.    ])

# 1, Just use the similarity value [-1, 1]. Discard the negative part, and use it as probability.


# Attention - difference between np.exp(x)/np.reciprocal(x) and np.exp(x,x)/np.reciprocal(x,x)
# 1, np.exp(x,x) changed the original value of x!!!
# 2, np.exp(x,x) x should be of ArrayType
x=np.array([2.1,1.3,4.2,3.5])
assert np.alltrue(np.exp(x) == np.exp(x,x))
assert np.alltrue(x == np.array([2.1,1.3,4.2,3.5]))
