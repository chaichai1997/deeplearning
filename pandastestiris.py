# -*- coding: utf-8 -*-
import os
import pandas as pd
import requests

PATH = r'./data/'
r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/iris'
                 '/iris.data')
with open(PATH + 'iris.data', 'w+') as f:
    f.write(r.text)

os.chdir(PATH)

# add title to csv files
df = pd.read_csv(PATH + 'iris.data', names=['sepal length', 'sepal width',
                                            'petal length', 'petal width', 'class'])

df.head()
