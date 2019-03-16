# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, discriminant_analysis, model_selection


"""
加载数据集，测试集大小为原始数据集的0.25倍 ，将数据切分为训练集与测试集
"""


def load_data():
    diabetes = datasets.load_diabetes()
    for i in range(0, 10):
        print(diabetes.target[i])
    return model_selection.train_test_split(diabetes.data, diabetes.target,
                                            test_size=0.25, random_state=0)


"""
标准线性回归
def __init__(self, fit_intercept=True, normalize=False, copy_X=True,
                 n_jobs=None）
        参数：
            fit_intercept：布尔值，指定是否计算b
            copy_X：布尔值，是否复制x
            n_jobs：任务并行CPU数目
        属性：
            coef_:Coefficient: 权重向量
            intercept_:intercept： b值
        方法：
            xx.fit()从训练集中学习
            xx.predict() 使用测试集测试
            xx.score()返回预测性能得分
data指定训练样本集、测试样本集、训练样本集对应的标签值、测试样本集对应的标签值
"""


def test_LinearRegression(*data):
    X_train, X_test, y_train, y_test = data
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    print('Coefficients:%s , intercept %.2f' % (regr.coef_, regr.intercept_))
    print("residual sum of squares: %.2f" % np.mean((regr.predict(X_test) - y_test)
                                                    ** 2))
    print('Score: %.2f' % regr.score(X_test, y_test))


"""
 岭回归,一种风格正则化方法，在损失函数中加入L2范数惩罚项
   (α||w||)平方的模
   __init__(self, alpha=1.0, fit_intercept=True, normalize=False,
                 copy_X=True, max_iter=None, tol=1e-3, solver="auto",
                 random_state=None):
    参数：
        alpha：正则化占比
        fit_intercept：布尔值，指定是否计算b
        copy_X：布尔值，是否复制x
        solver：指定最优化问题的解决算法
            auto：自动
            svd: 使用奇异值分解计算回归系数
        
        
   
"""


def test_Ridge(*args):
    X_train, X_test, y_train, y_test = args
    regr = linear_model.Ridge()
    regr.fit(X_train, y_train)
    print('Coefficients:%s , intercept %.2f' % (regr.coef_, regr.intercept_))
    print("residual sum of squares: %.2f" % np.mean((regr.predict(X_test) - y_test)
                                                    ** 2))
    print('Score: %.2f' % regr.score(X_test, y_test))


if __name__ == '__main__':
    X_train, X_test, y_train,  y_test = load_data()
    print("普通线性回归")
    test_LinearRegression(X_train, X_test, y_train,  y_test)
    print("岭回归")
    test_Ridge(X_train, X_test, y_train,  y_test)



