# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection


def load_data():
    iris = datasets.load_iris()
    X_train = iris.data
    y_train = iris.target
    return model_selection.train_test_split(X_train, y_train, test_size=0.25,
                                            random_state=0, stratify=y_train)


"""
决策树分类：DecisionTreeClassifier
_init__(self,
                 criterion="gini",  切分质量的评价准则 gini：基尼系数 entropy：熵
                 splitter="best",   指定切分原则，best/random
                 max_depth=None,     指定树的最大深度，整数
                 min_samples_split=2,  内部节点包含最少的样本数
                 min_samples_leaf=1,   叶子节点包含最少的样本数
                 min_weight_fraction_leaf=0., 叶子节点中最小权重系数
                 max_features=None,   指定了寻找best spilt时的最优特征数量
                 random_state=None,
                 max_leaf_nodes=None,    最大叶子节点数量
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 class_weight=None,   指定分类的权重
                 presort=False):   布尔值，指定是否需要提前排序数据
属性：
    classes_:分类的标签值
    feature_importances_:特征的重要程度
    max_features:max_features的推断值
    n_outputs_:执行fit后，输出的数量
    n_features_:执行fit后，特征的数量
    n_classes_:分类的数量
    tree_:底层决策树
方法：
    fit()
    score()
    predict()
    predict_log_prob(x)  x属于各个类别的概率值的对数
    predict_prob(x) x属于各个类别的概率值
    
"""


def test_DecisionTreeClassifier(*args):
    X_train, X_test, y_train, y_test =args
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    print("training score", clf.score(X_train, y_train))
    print("test score", clf.score(X_test, y_test))


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    test_DecisionTreeClassifier(X_train, X_test, y_train, y_test)
