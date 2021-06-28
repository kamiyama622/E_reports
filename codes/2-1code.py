# -*- coding: utf-8 -*-
"""
Created on Mon May 24 08:54:15 2021

@author: zawaz
"""
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

boston = load_boston()

boston_df = pd.DataFrame(boston.data, columns = boston.feature_names) 
boston_df['MEDV'] = boston.target

#相関関係の模索
cor_list = boston_df.corr()

#taxとindusの線形回帰モデルの作成
lr = LinearRegression()

X = boston_df[['INDUS']].values
Y = boston_df['TAX'].values

lr.fit(X, Y)

#plot
Y_pred = lr.predict(X)
fig = plt.figure(figsize=(15,10))
plt.plot(X, Y, 'r.')
plt.plot(X, Y_pred, 'k-')

plt.title('Regression Line')
plt.xlabel('Indus')
plt.ylabel('Tax')
plt.grid()

plt.show()

#評価
print('MSE: ', mean_squared_error(Y, Y_pred))

#課題
X1 = boston_df[['CRIM','RM']].values
Y1 = boston_df['MEDV'].values

lr.fit(X1, Y1)
print(lr.predict([[0.3,4]]))