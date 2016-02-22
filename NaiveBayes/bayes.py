# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 11:08:49 2016
bayes algorithm: classify a words as good or bad   [text classify]
@author: liudiwei
"""

import numpy as np

class NaiveBayes:
    def __init__(self):
        self._creteria = "NB"
        
    def _createVocabList(self, dataList):
        """
        创建一个词库向量
        """
        vocabSet = set([])
        for line in dataList:
            vocabSet = vocabSet | set(line)
        return list(vocabSet)
        
    #文档词集模型
    def _setOfWords2Vec(self, vocabList, inputSet):
        """
        功能:根据给定的一行词，将每个词映射到此库向量中，出现则标记为1，不出现则为0
        """
        outputVec = [0] * len(vocabList)
        for word in inputSet:
            if word in vocabList:
                outputVec[vocabList.index(word)] = 1
            else:
                print "the word:%s is not in my vocabulary!" % word
        return outputVec
    
        
    # 修改 _setOfWordsVec  文档词袋模型
    def _bagOfWords2VecMN(self, vocabList, inputSet):
        """
        功能：对每行词使用第二种统计策略，统计单个词的个数，然后映射到此库中
        输出：一个n维向量，n为词库的长度，每个取值为单词出现的次数
        """
        returnVec = [0]*len(vocabList)
        for word in inputSet:
            if word in vocabList:
                returnVec[vocabList.index(word)] += 1 # 更新此处代码
        return returnVec
    
    
    def _trainNB(self, trainMatrix, trainLabel):
        """
        输入：训练矩阵和类别标签,格式为numpy矩阵格式
        功能：计算条件概率和类标签概率
        """
        numTrainDocs = len(trainMatrix) #统计样本个数
        numWords = len(trainMatrix[0])  #统计特征个数，理论上是词库的长度
        pNeg = sum(trainLabel)/float(numTrainDocs) #计算负样本出现的概率
        
        p0Num = np.ones(numWords) #初始样本个数为1，防止条件概率为0，影响结果     
        p1Num = np.ones(numWords) #作用同上
        
        p0InAll = 2.0 #词库中只有两类，所以此处初始化为2(use laplace)
        p1InAll = 2.0 
        
        # 再单个文档和整个词库中更新正负样本数据
        for i in range(numTrainDocs):
            if trainLabel[i] == 1:
                p1Num += trainMatrix[i]
                p1InAll += sum(trainMatrix[i])
            else:
                p0Num += trainMatrix[i]
                p0InAll += sum(trainMatrix[i])
        
        #计算给定类别的条件下，词汇表中单词出现的概率
        #然后取log对数，解决条件概率乘积下溢
        p0Vect = np.log(p0Num/p0InAll) #计算类标签为0时的其它属性发生的条件概率
        p1Vect = np.log(p1Num/p1InAll)  #log函数默认以e为底  #p(ci|w=0)
        return p0Vect, p1Vect, pNeg
        
    def _classifyNB(self, vecSample, p0Vec, p1Vec, pNeg):
        """
        使用朴素贝叶斯进行分类,返回结果为0/1
        """
        prob_y0 = sum(vecSample * p0Vec) + np.log(1-pNeg)
        prob_y1 = sum(vecSample * p1Vec) + np.log(pNeg) #log是以e为底
        if prob_y0 < prob_y1:
            return 1
        else:
            return 0

    
    # 测试NB算法
    def testingNB(self, testSample):
        listOPosts, listClasses = loadDataSet()
        myVocabList = self._createVocabList(listOPosts)
        trainMat=[]
        for postinDoc in listOPosts:
            trainMat.append(self._bagOfWords2VecMN(myVocabList, postinDoc))
        p0V,p1V,pAb = self._trainNB(np.array(trainMat), np.array(listClasses))
        thisSample = np.array(self._bagOfWords2VecMN(myVocabList, testSample))
        result = self._classifyNB(thisSample, p0V, p1V, pAb)
        print testSample,'classified as: ', result
        return result


###############################################################################
def loadDataSet():
        wordsList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', ' and', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks','ate','my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
        classLable = [0,1,0,1,0,1] # 0：good; 1:bad
        return wordsList, classLable
        
if __name__=="__main__":    
    clf = NaiveBayes()
    testEntry = [['love', 'my', 'girl', 'friend'],
                 ['stupid', 'garbage'],
                 ['Haha', 'I', 'really', "Love", "You"],
                 ['This', 'is', "my", "dog"]]
    for item in testEntry:
        clf.testingNB(item)
