#!/usr/bin/env python      
# -*- coding: utf-8 -*-     
# Author: feng

from sklearn import svm

svm.LinearSVC.decision_function()
svm.LinearSVC._predict_proba_lr()

from sklearn import linear_model

linear_model.LogisticRegression.predict_proba()
linear_model.LogisticRegression._predict_proba_lr()
