# -*- coding:utf-8 -*-
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import  StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
import math
from sklearn import svm #svm导入
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.naive_bayes import GaussianNB

# ord_data = pd.read_csv("D:\\data\\001\\201807.csv")
# #数据标准化
ss=StandardScaler()

# gbdt=GradientBoostingRegressor(n_estimators=50,learning_rate=0.001,random_state=20)
# lr=LinearRegression()
# lrpol=LinearRegression()
# clf = svm.SVR(C=1,kernel='rbf',gamma=0.1)
# bg=BaggingRegressor(LinearRegression(),n_estimators=50,max_samples=0.001,max_features=0.7,random_state=20)
# ada=AdaBoostRegressor(LinearRegression(),n_estimators=50,learning_rate=0.001)
# gnb=GaussianNB()


# file_name = "E:\\cj\\march\\data\\001\\201807.csv"
# ord_data = pd.read_csv(file_name)
# train_data = ord_data.fillna(method="backfill")
# tem = ord_data['var001'].notnull()
# tem1 = pd.Series(['var001'])
# for k in range(68):
#     num_name = format(k + 1, '>3.0f').replace(" ", "0")
#     column_name = "var%s" % num_name
#     column = ord_data[column_name].notnull()
#     tem = tem & column
# all_know_data = ord_data[tem]
# X = all_know_data.drop(columns=['ts','wtid','var001'])
# Y = all_know_data['var001']
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# X_train = ss.fit_transform(X_train)
# X_test = ss.transform(X_test)


def MAverage( i, N):
    sum = 0
    for x in range(1, N+1):
        print(tem1[i-x])
        sum = sum + tem1[i-x]
    result = sum/N
    return result




file_name = "E:\\cj\\march\\data\\001\\201807.csv"
ord_data = pd.read_csv(file_name)
tem1 = ord_data['var001']
print(tem1[10])
s=len(tem1)
for i in range(s):
    if tem1[i] == 0:
        tem1[i] = tem1[i-1]+tem1[i-2]+tem1[i-3]
        tem1[i] = tem1[i]/3
        print(tem1[i])




# start_Line = time.clock()
# lr.fit(X_train, Y_train)
# Y_test_Line = lr.predict(X_test)
# mse_Line=np.sum(math.e**-((abs(Y_test-Y_test_Line)*100)/Y_test))
# print ('Line损失：' ,mse_Line)
# print ('Line耗时：' ,(time.clock() - start_Line))


# start_Bagging = time.clock()
# bg.fit(X_train, Y_train)
# Y_test_Bagging = bg.predict(X_test)
# mse_Bagging=np.sum(math.e**-((abs(Y_test-Y_test_Bagging)*100)/Y_test))
# print ('Bagging损失：' ,mse_Bagging)
# print ('Bagging耗时：' ,(time.clock() - start_Bagging))
#
#
# start_AdaBoost = time.clock()
# ada.fit(X_train, Y_train)
# Y_test_AdaBoost = ada.predict(X_test)
# mse_AdaBoost=np.sum(math.e**-((abs(Y_test-Y_test_AdaBoost)*100)/Y_test))
# print ('AdaBoost损失：' ,mse_AdaBoost)
# print ('AdaBoost耗时：' ,(time.clock() - start_AdaBoost))


# start_SVM = time.clock()
# clf.fit(X_train, Y_train)
# Y_test_SVM = lr.predict(X_test)
# mse_Line=np.sum(math.e**-((abs(Y_test-Y_test_SVM)*100)/Y_test))
# print ('SVM损失：' ,mse_Line)
# print ('SVM耗时：' ,(time.clock() - start_SVM))

# start_LinePol = time.clock()
# pol = PolynomialFeatures(degree = 2)
# xtrain_pol = pol.fit_transform(X_train)
# xtest_pol = pol.fit_transform(X_test)
# lrpol.fit(xtrain_pol, Y_train)
# Y_test_LinePol = lr.predict(xtest_pol)
# mse_Line=np.sum(math.e**-((abs(Y_test-Y_test_LinePol)*100)/Y_test))
# print ('LinePol损失：' ,mse_Line)
# print ('LinePol耗时：' ,(time.clock() - start_LinePol))





