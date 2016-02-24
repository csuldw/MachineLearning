# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 14:41:04 2016
Hand Writing
@author: liudiwei
"""

import os
import numpy as np
import KNN as knn

def imgToVector(filename):
    returnVect = np.zeros((1, 1024))
    fo = open(filename, 'r')
    fr = fo.readlines()
    m = len(fr)
    for i in range(m):
        oneline = fr[i].strip()
        n = len(oneline)
        if not n:
            continue
        for j in range(n):
            returnVect[0,n*i+j] = int(oneline[j])  #m行n列  当前元素为n*i+j
    return returnVect

#加载数据集
def loadDataset(filedir):
    trainlist = os.listdir(filedir)
    m = len(trainlist)
    data_X = np.zeros((m,1024))
    data_y = []
    for i in range(len(trainlist)):
        filename = trainlist[i]
        label_i = int(filename.split('_')[0])
        data_y.append(label_i)
        oneSample_X = imgToVector(filedir + "/" + trainlist[i])
        data_X[i,:]  = oneSample_X
    return data_X, data_y
    
#调用KNN.py中的KNN算法，并统计数据的错误率
def handwritingClassTest(train_X, train_y, test_X, test_y):
    clf = knn.KNNClassifier(k=3)
    len_test = len(test_y)
    error_count = 0.0
    results = clf.classify(test_X, train_X, train_y)
    arr_res = np.array(results) - np.array(test_y)
    for elem in arr_res:
        error_count += np.abs(elem) 
    error_rate = error_count/len_test
    print error_rate
    return error_rate

def main():
    traindir = "data/trainingDigits"
    testdir = "data/testDigits"
    train_X, train_y = loadDataset(traindir)
    test_X, test_y = loadDataset(testdir)
    handwritingClassTest(train_X, train_y, test_X, test_y)

if __name__=="__main__":
    main()