# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 12:17:10 2016
Decision Tree Source Code
@author: liudiwei
"""
import os
import numpy as np

class DecitionTree():
    """this is a decision tree classifier. """
    
    def __init__(self, criteria='C4.5'):
        self._tree = None
        if criteria == 'ID3' or criteria == 'C4.5':
            self._criteria = criteria
        else:
            raise Exception("criterion should be ID3 or C4.5")
    
    def _calEntropy(slef, y):
        '''
        _calEntropy用于计算香农熵 e=-sum(pi*log pi)
        其中y为数组array
        输出entropy
        '''
        n = y.shape[0]  
        labelCounts = {}
        for label in y:
            if label not in labelCounts.keys():
                labelCounts[label] = 1
            else:
                labelCounts[label] += 1
        entropy = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key])/n
            entropy -= prob * np.log2(prob)
        return entropy
    
    def _splitData(self, X, y, axis, cutoff):
        """
        该函数返回数据集中特征下标为axis，特征值等于cutoff的子数据集
        """
        ret = []
        featVec = X[:,axis]
        n = X.shape[1]      #特征个数
        X = X[:,[i for i in range(n) if i!=axis]]
        for i in range(len(featVec)):
            if featVec[i] == cutoff:
                ret.append(i)
        return X[ret, :], y[ret]   
            
    def _chooseBestSplit(self, X, y):
        """ID3 & C4.5
        根据信息增益或者信息增益率来获取最好的划分特征
        """
        numFeat = X.shape[1]
        baseEntropy = self._calEntropy(y)
        bestSplit = 0.0
        best_idx  = -1
        for i in range(numFeat):
            featlist = X[:,i]   #得到第i个特征对应的特征列
            uniqueVals = set(featlist)
            curEntropy = 0.0
            splitInfo = 0.0
            for value in uniqueVals:
                sub_x, sub_y = self._splitData(X, y, i, value)
                prob = len(sub_y)/float(len(y))      #计算某个特征的某个值的概率
                curEntropy += prob * self._calEntropy(sub_y)    #迭代计算条件熵
                splitInfo -=  prob * np.log2(prob) #分裂信息，用于计算信息增益率
            IG = baseEntropy - curEntropy
            if self._criteria=="ID3":
                if IG > bestSplit:
                    bestSplit = IG
                    best_idx = i
            if self._criteria=="C4.5":
                if splitInfo == 0.0:
                    pass
                IGR = IG/splitInfo
                if IGR > bestSplit:
                    bestSplit = IGR
                    best_idx = i
        return best_idx
        
    def _majorityCnt(self, labellist):
        """
        返回labellist中出现次数最多的label
        """
        labelCount={}
        for vote in labellist:
            if vote not in labelCount.keys(): labelCount[vote] = 0
            labelCount[vote] += 1
        sortedClassCount = sorted(labelCount.iteritems(), key=lambda x:x[1], \
                                     reverse=True)
        return sortedClassCount[0][0]

    def _createTree(self, X, y, featureIndex):
        labelList = list(y)
        if labelList.count(labelList[0]) == len(labelList): 
            return labelList[0]
        if len(featureIndex) == 0:
            return self._majorityCnt(labelList)
        bestFeatIndex = self._chooseBestSplit(X,y)
        bestFeatAxis = featureIndex[bestFeatIndex]
        featureIndex = list(featureIndex)
        featureIndex.remove(bestFeatAxis)
        featureIndex = tuple(featureIndex)
        myTree = {bestFeatAxis:{}}
        featValues = X[:, bestFeatIndex]
        uniqueVals = set(featValues)
        for value in uniqueVals:
            #对每个value递归地创建树
            sub_X, sub_y = self._splitData(X,y, bestFeatIndex, value)
            myTree[bestFeatAxis][value] = self._createTree(sub_X,sub_y,\
                                            featureIndex)
        return myTree  
        
    def fit(self, X, y):
        #对数据X和y进行类型检测，保证其为array
        if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
            pass
        else:
            try:
                X = np.array(X)
                y = np.array(y)
            except:
                raise TypeError("numpy.ndarray required for X,y")
        featureIndex  = tuple(['x'+str(i) for i in range(X.shape[1])])
        self._tree = self._createTree(X,y,featureIndex)
        return self  #allow using: clf.fit().predict()
        
    def _classify(self, tree, sample):
        featIndex = tree.keys()[0]
        secondDict = tree[featIndex]
        key = sample[int(featIndex[1:])]
        valueOfkey = secondDict[key]
        if type(valueOfkey).__name__=='dict':
            return self._classify(valueOfkey, sample)
        else: 
            return valueOfkey
        
    def predict(self, X):
        if self._tree==None:
            raise NotImplementedError("Estimator not fitted, call `fit` first")
        if isinstance(X,np.ndarray): 
            pass
        else: 
            try:
                X = np.array(X)
            except:
                raise TypeError("numpy.ndarray required for X")
            
        if len(X.shape)==1:
            return self._classify(self._tree, X)
        else:
            result = []
            for i in range(X.shape[0]):
                result.append(self._classify(self._tree, X[i]))
            return np.array(result)

    def show(self):
        if self._tree==None:
            pass
        #plot the tree using matplotlib
        import treePlotter
        treePlotter.createPlot(self._tree)
        
    
if __name__=="__main__":
    trainfile=r"data\train.txt"
    testfile=r"data\test.txt"
    import sys
    sys.path.append(r"F:\CSU\Github\MachineLearning\lib")  
    import dataload as dload
    train_x, train_y = dload.loadData(trainfile)
    test_x, test_y = dload.loadData(testfile)
    clf = DecitionTree()
    clf.fit(train_x, train_y)
    result = clf.predict(test_x)    
    clf.show()
    