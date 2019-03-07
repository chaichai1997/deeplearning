# -*- coding: utf-8 -*-
import os
import pandas as pd
import requests
import matplotlib.pyplot as plt
PATH = r'E:/deep learning/code/data/'
r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/iris'
                 '/iris.data')
with open(PATH + 'iris.data', 'w') as f:
    f.write(r.text)

os.chdir(PATH)
print(PATH)
# add title to csv files
df = pd.read_csv(PATH + 'iris.data',
                 names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'],
                 engine='python')  # 添加标题并读取前五行
print(df.head())
s = df['sepal length']   # 获取sepal length 列
print(df.ix[:3, :2])
print(df['class'].unique())
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(df['petal width'], color='black')
ax.set_ylabel('count', fontsize=12)
ax.set_xlabel('wount', fontsize=12)
plt.title('iris petal width',fontsize =14, y=1.01)
plt.show()