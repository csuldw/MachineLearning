# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 15:10:42 2016

@author: liudiwei
"""

import numpy as np
from adaboost import AdaboostClassifier

#每行数据以\t隔开，最后一列为类标号
def loadDataSet(datafile):
    featData = []; labelDate = []
    with open(datafile, 'r') as fr_file:
        for eachline in fr_file:
            oneline = eachline.split('\t')
            tempArr = []
            for i in range(len(oneline)-1):
                tempArr.append(float(oneline[i]))
            featData.append(tempArr)
            labelDate.append(int(float(oneline[-1].strip())))
    return featData, labelDate   #返回的数据是list

def claErrorRate(results, test_y):
    len_test = len(test_y)
    error_count = 0.0
    label_pred = np.mat(results) 
    label_true = np.mat(test_y).transpose()
    arr_res = label_pred - label_true
    for elem in arr_res:
        error_count += np.abs(elem) 
    error_rate = error_count/len_test
    print error_rate

def main():
    trainfile=r"data\train.txt"
    testfile=r"data\test.txt"
    train_X, train_y = loadDataSet(trainfile)
    test_X, test_y = loadDataSet(testfile)
    clf = AdaboostClassifier(max_iter=100)
    classifier = clf.fit(train_X, train_y)
    results = clf.predict(test_X, classifier)
    claErrorRate(results, test_y)

if __name__=="__main__":
    result = main()
    