# -*- coding: utf-8 -*-
import os
import pandas as pd
import request

PATH = r'E:\deep learning\code\data'
r = request.get('https://archive.ics.uci.edu/ml/machine-learning-databases/iris'
                 '/iris.data')
with open(PATH + 'iris.data', 'w') as f:
    f.write(r.text)

os.chdir(PATH)

# add title to csv files
df = pd.read_csv(PATH + 'iris.data',
                 names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'],
                 engine='python')

print(df.head())
print(df['class'].unique())

