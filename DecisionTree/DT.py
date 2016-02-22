# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 12:17:10 2016
Decision Tree Source Code
@author: liudiwei
"""
import os
import numpy as np

class DecitionTree():
    """This is a decision tree classifier. """
    
    def __init__(self, criteria='ID3'):
        self._tree = None
        if criteria == 'ID3' or criteria == 'C4.5':
            self._criteria = criteria
        else:
            raise Exception("criterion should be ID3 or C4.5")
    
    def _calEntropy(slef, y):
        '''
        功能：_calEntropy用于计算香农熵 e=-sum(pi*log pi)
        参数：其中y为数组array
        输出：信息熵entropy
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
        参数：X为特征,y为label,axis为某个特征的下标,cutoff是下标为axis特征取值值
        输出：返回数据集中特征下标为axis，特征值等于cutoff的子数据集
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
        参数：X为特征，y为label
        功能：根据信息增益或者信息增益率来获取最好的划分特征
        输出：返回最好划分特征的下标
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
        参数:labellist是类标签，序列类型为list
        输出：返回labellist中出现次数最多的label
        """
        labelCount={}
        for vote in labellist:
            if vote not in labelCount.keys(): 
                labelCount[vote] = 0
            labelCount[vote] += 1
        sortedClassCount = sorted(labelCount.iteritems(), key=lambda x:x[1], \
                                     reverse=True)
        return sortedClassCount[0][0]

    def _createTree(self, X, y, featureIndex):
        """
        参数:X为特征,y为label,featureIndex类型是元组，记录X特征在原始数据中的下标
        输出:根据当前的featureIndex创建一颗完整的树
        """
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
            myTree[bestFeatAxis][value] = self._createTree(sub_X, sub_y, \
                                            featureIndex)
        return myTree  
        
    def fit(self, X, y):
        """
        参数：X是特征，y是类标签
        注意事项：对数据X和y进行类型检测，保证其为array
        输出：self本身
        """
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
        """
        用训练好的模型对输入数据进行分类 
        注意：决策树的构建是一个递归的过程，用决策树分类也是一个递归的过程
        _classify()一次只能对一个样本（sample）分类
        """
        featIndex = tree.keys()[0] #得到数的根节点值
        secondDict = tree[featIndex] #得到以featIndex为划分特征的结果
        axis=featIndex[1:] #得到根节点特征在原始数据中的下标
        key = sample[int(axis)] #获取待分类样本中下标为axis的值
        valueOfKey = secondDict[key] #获取secondDict中keys为key的value值
        if type(valueOfKey).__name__=='dict': #如果value为dict，则继续递归分类
            return self._classify(valueOfKey, sample)
        else: 
            return valueOfKey
        
    def predict(self, X):
        if self._tree==None:
            raise NotImplementedError("Estimator not fitted, call `fit` first")
        #对X的类型进行检测，判断其是否是数组
        if isinstance(X, np.ndarray): 
            pass
        else: 
            try:
                X = np.array(X)
            except:
                raise TypeError("numpy.ndarray required for X")
            
        if len(X.shape) == 1:
            return self._classify(self._tree, X)
        else:
            result = []
            for i in range(X.shape[0]):
                value = self._classify(self._tree, X[i])
                print str(i+1)+"-th sample is classfied as:", value 
                result.append(value)
            return np.array(result)

    def show(self, outpdf):
        if self._tree==None:
            pass
        #plot the tree using matplotlib
        import treePlotter
        treePlotter.createPlot(self._tree, outpdf)
    
if __name__=="__main__":
    trainfile=r"data\train.txt"
    testfile=r"data\test.txt"
    import sys
    sys.path.append(r"F:\CSU\Github\MachineLearning\lib")  
    import dataload as dload
    train_x, train_y = dload.loadData(trainfile)
    test_x, test_y = dload.loadData(testfile)
    
    clf = DecitionTree(criteria="C4.5")
    clf.fit(train_x, train_y)
    result = clf.predict(test_x)    
    outpdf = r"tree.pdf"
    clf.show(outpdf)
    