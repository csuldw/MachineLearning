# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Created on Sat Oct 04 15:22:55 2015
data load
@author: liudiwei
"""
import csv
import numpy as np


'''
参数：
    - datafile: 一个输入文件，文件格式如下    
        每行数据以\t隔开，最后一列为类标号
        格式：[data1]\t[data2]\t[data3]\t...\tdata[n]\t[classlabel\n]
返回结果：
    - featData：特征数据，二维list存储
    - labelData：类别标签，一维list存储
'''
def loadData(datafile):
    featData = []; labelDate = []
    with open(datafile, 'r') as fr_file:
        for eachline in fr_file:
            oneline = eachline.split('\t')
            tempArr = []
            for i in range(len(oneline)-1):
                value = int(float("%.2f" %float(oneline[i])))
                tempArr.append(value)
            featData.append(tempArr)
            labelDate.append(int(float(oneline[-1].strip())))
    return featData, labelDate   #返回的数据是list

#将数组中的数据全部转化为int类型
def toInt(array):
    array = np.mat(array)
    m, n = np.shape(array)
    newArray=np.zeros((m,n)) #分配一个新的m行n列的零数组
    for i in xrange(m):
        for j in xrange(n):
            newArray[i,j] = int(array[i,j])
    return newArray

#归一化
def normalizing(array):
    m, n = np.shape(array)
    for i in xrange(m):
        for j in xrange(n):
            if array[i,j] != 0:
                array[i,j] = 1
    return array

'''
解释：
    - 加载csv格式数据,格式如data目录下的train.csv
    - 返回的是m*n的特征以及1*mlabel两个数组 
'''
def loadCSVfile1(datafile):
    filelist = []
    with open(datafile) as file:
        lines = csv.reader(file)
        for oneline in lines:
            filelist.append(oneline)
    filelist.remove(filelist[0])
    filelist = np.array(filelist)
    label = filelist[:,0]      # m*1
    data = filelist[:, 1:]     # m*n
    return normalizing(toInt(data)), np.transpose(toInt(label)) 

#使用numpy自带的csv加载函数
def loadCSVfile2(datafile):
    row_data = np.loadtxt(datafile, dtype=np.str, delimiter=",")
    train_data = row_data[1:,1:].astype(np.float)   #m*n
    label = row_data[1:,0].astype(np.float)         #m*1
    return normalizing(toInt(train_data)), np.transpose(toInt(label))

    
if __name__=="__main__":
    datafile = "data/train.csv"
    data3,data4 = loadCSVfile1(datafile) #最后返回的是m*n的特征以及m*1的类标号