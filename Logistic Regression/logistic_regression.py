# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Created on Sat Feb 20 20:02:08 2016
Logistic Regression
@author: liudiwei
"""
import numpy as np
    
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

#定义一个sigmoid函数
def sigmoid(fx):
    return 1.0/(1+np.exp(-fx))

#alpha为步长（学习率）；maxCycles最大迭代次数
def gradDescent(featData, labelData, alpha, maxCycles):
    dataMat = np.mat(featData)                      #size: m*n
    labelMat = np.mat(labelData).transpose()        #size: m*1
    m, n = np.shape(dataMat)
    weigh = np.ones((n, 1)) 
    for i in range(maxCycles):
        hx = sigmoid(dataMat * weigh)
        error = labelMat - hx       #size:m*1
        weigh = weigh + alpha * dataMat.transpose() * error#根据误差修改回归系数
    return weigh

#使用梯度下降方法训练模型，如果使用其它的寻参方法，此处可以做相应修改
def trainLogRegres(train_x, train_y, alpha=0.01, maxCycles=100):
    return gradDescent(train_x, train_y, alpha, maxCycles)

#使用学习得到的参数进行分类
def classify(testfile, weigh):
    testSet, testLabel = loadDataSet(testfile)
    dataMat = np.mat(testSet)
    labelMat = np.mat(testLabel).transpose()  #使用transpose()转置
    hx = sigmoid(dataMat*weigh)  #size:m*1
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

#主函数
if __name__=="__main__":
    trainfile=r"data\train.txt"
    testfile=r"data\test.txt"
    trainSet, trainLabel = loadDataSet(trainfile)
    weigh = trainLogRegres(trainSet, trainLabel, alpha=0.01, maxCycles=500)
    classify(testfile, weigh)