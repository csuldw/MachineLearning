# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Created on Sat Feb 20 20:02:08 2016
Logistic Regression
@author: liudiwei
"""
import numpy as np


class LogisticRegressionClassifier():
    
    def __init__(self):
        self._alpha = None
    

    #定义一个sigmoid函数
    def _sigmoid(self, fx):
        return 1.0/(1 + np.exp(-fx))

    #alpha为步长（学习率）；maxCycles最大迭代次数
    def _gradDescent(self, featData, labelData, alpha, maxCycles):
        dataMat = np.mat(featData)                      #size: m*n
        labelMat = np.mat(labelData).transpose()        #size: m*1
        m, n = np.shape(dataMat)
        weigh = np.ones((n, 1)) 
        for i in range(maxCycles):
            hx = self._sigmoid(dataMat * weigh)
            error = labelMat - hx       #size:m*1
            weigh = weigh + alpha * dataMat.transpose() * error#根据误差修改回归系数
        return weigh

    #使用梯度下降方法训练模型，如果使用其它的寻参方法，此处可以做相应修改
    def fit(self, train_x, train_y, alpha=0.01, maxCycles=100):
        return self._gradDescent(train_x, train_y, alpha, maxCycles)

    #使用学习得到的参数进行分类
    def predict(self, test_X, test_y, weigh):
        dataMat = np.mat(test_X)
        labelMat = np.mat(test_y).transpose()  #使用transpose()转置
        hx = self._sigmoid(dataMat*weigh)  #size:m*1
        m = len(hx)
        error = 0.0
        for i in range(m):
            if int(hx[i]) > 0.5:
                print str(i+1)+'-th sample ', int(labelMat[i]), 'is classfied as: 1' 
                if int(labelMat[i]) != 1:
                    error += 1.0
                    print "classify error."
            else:
                print str(i+1)+'-th sample ', int(labelMat[i]), 'is classfied as: 0' 
                if int(labelMat[i]) != 0:
                    error += 1.0
                    print "classify error."
        error_rate = error/m
        print "error rate is:", "%.4f" %error_rate
        return error_rate

