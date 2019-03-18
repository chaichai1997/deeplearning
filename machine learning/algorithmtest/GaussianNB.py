# -*- coding: utf-8 -*-
from sklearn import datasets, model_selection, naive_bayes
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    digit = datasets.load_digits()
    return model_selection.train_test_split(digit.data, digit.target,
                                            test_size=0.25, random_state=0)

"""
高斯贝叶斯分类器
"""


def test_GaussianNB(*args):
    X_train, X_test, y_train, y_test = args
    cls = naive_bayes.GaussianNB()
    cls.fit(X_train, y_train)
    print('training score:', cls.score(X_train, y_train))
    print('testing score:', cls.score(X_test, y_test))


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    test_GaussianNB(X_train, X_test, y_train, y_test)

