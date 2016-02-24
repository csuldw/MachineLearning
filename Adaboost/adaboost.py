# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 08:57:10 2016
Adaboost algorithm
@author: liudiwei
"""
import numpy as np


class AdaboostClassifier(object):
    
    def __init__(self, max_iter=100, numSteps=10, num_iter=50):
        self._max_iter = max_iter
        self._numSteps = numSteps
        self._num_iter = num_iter
        
    def _stumpClassify(self, X, axis, threshVal, threshIneq):
        """
        功能：一个单层决策树，通过阈值比较对数据集进行分类
        返回：
        """
        retArray = np.ones((np.shape(X)[0], 1)) #生成一个m*1的矩阵，全1
        if threshIneq == 'lt':  #设定规则为"lt"，小于阈值则为-1.0
            retArray[X[:, axis] <= threshVal] = -1.0
        else:
            retArray[X[:, axis] > threshVal] = -1.0
        return retArray
        
    def _buildStump(self, X, y, D):
        """
        输入：X, y和初始化的权重值D
        功能：找到最佳的单层决策树
        返回：一个字典(dim, ineq, thresh)、错误率、基于该特征和阈值下的类别估计值
        """
        dataMat = np.mat(X)
        labelMat = np.mat(y).T #类标签转置
        m, n = np.shape(dataMat)  #获得矩阵的维数
        bestStump = {} 
        bestClasEst = np.mat(np.zeros((m,1)))
        minError = np.inf #初始化最小误差，令其为最大值
        for i in range(n):#遍历所有的维数
            minVal = dataMat[:,i].min()
            maxVal = dataMat[:,i].max()
            stepSize = (maxVal-minVal)/self._numSteps
            for j in range(-1, int(self._numSteps)+1):
                for inequal in ['lt', 'gt']: 
                    threshVal = (minVal + float(j)*stepSize)
                    predVals = self._stumpClassify(dataMat,i,threshVal,inequal)
                    errArr = np.mat(np.ones((m,1)))  
                    errArr[predVals == labelMat] = 0 #统计误分类的个数
                    weightedError = D.T*errArr  #计算加权错误率
                    if weightedError < minError:
                        minError = weightedError
                        bestClasEst = predVals.copy()
                        bestStump['dim'] = i
                        bestStump['thresh'] = threshVal
                        bestStump['ineq'] = inequal
        return bestStump, minError, bestClasEst
    
    
    def fit(self, X, y):
        """
        输入：X，y，迭代次数
        输出：弱分类器weakClassArr（字典）
        """
        weakClassArr = []
        m = np.shape(X)[0]
        D = np.mat(np.ones((m, 1))/m)   #初始化权重值都相等，为1/m
        aggClassEst = np.mat(np.zeros((m, 1))) #样本点类别估计累计值
        for i in range(self._num_iter):
            bestStump, error, classEst = self._buildStump(X, y, D) 
            alpha = float(0.5 * np.log((1.0-error) / max(error, 1e-16)))
            bestStump['alpha'] = alpha  
            weakClassArr.append(bestStump)                  
            expon = np.multiply(-1 * alpha * np.mat(y).T, classEst) 
            D = np.multiply(D, np.exp(expon))                              
            D = D/D.sum()
            #计算所有分类器的训练误差，如果为0则提前结束循环
            aggClassEst += alpha*classEst
            aggErrors = np.multiply(\
                            np.sign(aggClassEst) != np.mat(y).T,np.ones((m,1)))
            errorRate = aggErrors.sum()/m
            #print "total error is: ", errorRate
            if errorRate == 0.0: 
                break
        return weakClassArr
    
    def predict(self, test_X, classifierArr):
        """
        输入: 测试集和分类器
        输出：分类结果（二分类）
        """
        dataMat = np.mat(test_X) 
        m = np.shape(dataMat)[0]
        aggClassEst = np.mat(np.zeros((m,1)))
        for i in range(len(classifierArr)):
            classEst = self._stumpClassify(dataMat,\
                                        classifierArr[i]['dim'],\
                                        classifierArr[i]['thresh'],\
                                        classifierArr[i]['ineq'])
            aggClassEst += classifierArr[i]['alpha'] * classEst
            #print aggClassEst
        return np.sign(aggClassEst)