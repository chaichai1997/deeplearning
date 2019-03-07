# -*- coding: utf-8 -*-
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import  StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

# ord_data = pd.read_csv("D:\\data\\001\\201807.csv")
# #数据标准化
ss=StandardScaler()
lr=LinearRegression()

ENUM_COL = ["var016", "var020", "var047"]
BOOL_COL = ["var053", "var066"]
DOUBLE_COL_TWO=["var001","var003","var005","var006","var010","var011","var012","var013","var017","var019","var021","var022","var025","var026","var027","var028","var029","var030","var031","var033","var034","var037","var038","var039","var044","var048","var049","var052","var056","var057","var058","var059","var060","var061","var063","var064","var067","var068"]
DOUBLE_COL_ONE=["var002","var007","var014","var015","var018","var024","var035","var036","var040","var045","var051","var055","var062","var065"]
DOUBLE_COL=["var004","var008","var009","var023","var032","var041","var042","var043","var046","var050","var054"]

def fillOneClounm(num,td):
    var_num=format(num, '>3.0f').replace(" ", "0")
    var_num="var%s" % var_num


    X = all_know_data.drop(columns=['ts','wtid',var_num])
    Y = all_know_data[var_num]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)
    lr.fit(X_train, Y_train)

    unknown_data = ord_data[ord_data[var_num].isnull()]
    train_data_ready = td[ord_data[var_num].isnull()]
    train_data_ready_matrix = train_data_ready.drop(columns=['ts','wtid',var_num]).as_matrix()

    predictedResult = lr.predict(train_data_ready_matrix)
    # 用得到的预测结果填补原缺失数据
    ord_data.loc[(ord_data[var_num].isnull()), var_num] = predictedResult

start_time = time.time()
submit = pd.read_csv("E:\\cj\\march\\template_submit_result.csv")
result = pd.DataFrame()
for i in range(1,34):#for i in range(1,34):
    print("第%d个文件"%i)
    num_name = format(i, '>3.0f').replace(" ", "0")
    file_name = "E:\\cj\\march\\data\\%s\\201807.csv"%num_name
    ord_data = pd.read_csv(file_name)
    train_data = ord_data.fillna(method="backfill")
    tem = ord_data['var001'].notnull()
    for k in range(68):
        num_name = format(k + 1, '>3.0f').replace(" ", "0")
        column_name = "var%s" % num_name
        column = ord_data[column_name].notnull()
        tem = tem & column
    all_know_data = ord_data[tem]
    for j in range(68):
        fillOneClounm(j + 1,train_data)
        print('完成%s列' % str(j + 1))

    temp_submit = submit[submit.wtid == i][["ts", "wtid"]]
    temp_submit = pd.merge(temp_submit, ord_data, on=["ts", "wtid"], how="left")
    if i == 1:
        result = temp_submit
    else:
        result = pd.concat([result, temp_submit])

if len(submit) == len(result):
    result.fillna(method="backfill", inplace=True)
    result[ENUM_COL] = result[ENUM_COL].astype(int)
    result[BOOL_COL] = result[BOOL_COL].astype(int)
    result[DOUBLE_COL] = result[DOUBLE_COL].astype(int)
    result[DOUBLE_COL_TWO] = result[DOUBLE_COL_TWO].round(decimals=2)
    result[DOUBLE_COL_ONE] = result[DOUBLE_COL_ONE].round(decimals=1)
    result.to_csv("./2019022801_result.csv", index=False)
else:
    print(result)