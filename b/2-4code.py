# -*- coding: utf-8 -*-
"""
Created on Sun May 30 18:55:59 2021

@author: zawaz
"""
# %matplotlib inline
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
import seaborn
warnings.simplefilter('ignore')

data_breast_cancer = load_breast_cancer()
X = pd.DataFrame(data_breast_cancer["data"],columns=data_breast_cancer["feature_names"])
y = pd.DataFrame(data_breast_cancer["target"],columns=["target"])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

lr = LogisticRegressionCV(cv=10, random_state=0)
lr.fit(X_train_std,y_train)
print('train score:',lr.score(X_train_std,y_train))
print('test score:',lr.score(X_test_std,y_test))

# plt.figure(figsize=(20, 20))
# seaborn.heatmap(pd.DataFrame(X_train_std).corr(), annot=True)

pca = PCA(n_components=2)
pca.fit(X_train_std)
# plt.bar([n for n in range(1, len(pca.explained_variance_ratio_)+1)], pca.explained_variance_ratio_)

X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.fit_transform(X_test_std)

lr = LogisticRegressionCV(cv=10, random_state=0)
lr.fit(X_train_pca,y_train)
print('train score:',lr.score(X_train_pca,y_train))
print('test score:',lr.score(X_test_pca,y_test))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1],c = y_train.values.flatten(),edgecolor = 'black',cmap='Blues')