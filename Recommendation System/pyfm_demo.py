# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 23:49:24 2019

pyfm install：将https://github.com/coreylynch/pyFM 下载到本地，去掉setup.py里面的
libraries=["m"],然后安装即可.

@author: liudiwei
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer

def load_data():
    """
    调用sklearn的iris数据集，筛选正负样本并构造切分训练测试数据集
    """
    iris_data = load_iris()
    X = iris_data['data']
    y = iris_data['target'] == 2
    data = [ {v: k for k, v in dict(zip(i, range(len(i)))).items()}  for i in X]
    
    X_train,X_test,y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=0)
    return X_train,X_test,y_train, y_test

X_train,X_test,y_train, y_test = load_data()

v = DictVectorizer()
X_train = v.fit_transform(X_train)
X_test = v.transform(X_test)

fm = pylibfm.FM(num_factors=2, 
                num_iter=200, 
                verbose=True, 
                task="classification", 
                initial_learning_rate=0.001, 
                learning_rate_schedule="optimal")

fm.fit(X_train, y_train)

y_preds = fm.predict(X_test)
y_preds_label = y_preds > 0.5
from sklearn.metrics import log_loss,accuracy_score
print ("Validation log loss: %.4f" % log_loss(y_test, y_preds))
print ("accuracy: %.4f" % accuracy_score(y_test, y_preds_label))

