# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 16:04:30 2016

@author: liudiwei
"""

from logistic_regression import LogisticRegressionClassifier

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
    


def main():
    trainfile = r"data\train.txt"
    testfile = r"data\test.txt"
    train_X, train_y = loadDataSet(trainfile)
    test_X, test_y = loadDataSet(testfile)
    clf = LogisticRegressionClassifier()
    weigh = clf.fit(train_X, train_y, alpha=0.01, maxCycles=500)
    clf.predict(test_X, test_y, weigh)
    
#主函数
if __name__=="__main__":
    main()