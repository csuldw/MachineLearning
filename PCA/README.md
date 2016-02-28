---
layout: post
date: 2015-11-03 16:24
title: "PCA主成分分析Python实现"
categories: ML
tag: 
	- Machine Learning
	- PCA
comment: true
---


PCA（principle component analysis） 翻译过来叫主成分分析，主要是降低数据集的维度，然后挑选出主要的特征。今天自己跟着下面这篇文章的步骤，把PCA用python实现了一遍，具体的思想可以参考这篇文章，讲的通俗易懂，主要是有个实例参考，值得拥有！

- [JerryLead之PCA主成分分析](http://www.cnblogs.com/jerrylead/archive/2011/04/18/2020209.html)

下面自己来简单的清理下思路！

## PCA思想

 移动坐标轴，将n维特征映射到k维上（k<n），这k维是全新的正交特征。这k维特征称为主元，是重新构造出来的k维特征，而不是简单地从n维特征中去除其余n-k维特征。

说到PCA难免会提到LDA（linear discriminate analysis，线性判别分析），以及FA（factor analysis，因子分析）。关于LDA，打算有时间也用代码实现一遍，下面给出它的主要思想。

LDA思想：最大类间距离，最小类内距离。简而言之，第一，为了实现投影后的两个类别的距离较远，用映射后两个类别的均值差的绝对值来度量。第二，为了实现投影后，每个类内部数据点比较聚集，用投影后每个类别的方差来度量。

三者的描述如下

 以下内容引自 [Wikipedia- Linear discriminant analysis](https://en.wikipedia.org/wiki/Linear_discriminant_analysis)
>LDA is also closely related to principal component analysis (PCA) and factor 		analysis in that they both look for linear combinations of variables which best explain the data.[4] LDA explicitly attempts to model the difference between the classes of data. PCA on the other hand does not take into account any difference in class, and factor analysis builds the feature combinations based on differences rather than similarities. Discriminant analysis is also different from factor analysis in that it is not an interdependence technique: a distinction between independent variables and dependent variables (also called criterion variables) must be made.

区别：PCA选择样本点投影具有最大方差的方向，LDA选择分类性能最好的方向。

好了，下面来看下实现源码！

## 基本步骤


基本步骤：

- 对数据进行归一化处理（代码中并非这么做的，而是直接减去均值）
- 计算归一化后的数据集的协方差矩阵                   
- 计算协方差矩阵的特征值和特征向量
- 保留最重要的k个特征（通常k<n），可以自己制定，也可以选择个阈值，让后通过前k个特征值之和减去后面n-k个特征值之和大于这个阈值，找到这个k
- 找出k个特征值对应的特征向量
- 将m * n的数据集乘以k个n维的特征向量的特征向量（n * k）,得到最后降维的数据。

## 源码实现

1.首先引入numpy，由于测试中用到了pandas和matplotlib，所以这里一并加载

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

2.定义一个均值函数

```
#计算均值,要求输入数据为numpy的矩阵格式，行表示样本数，列表示特征    
def meanX(dataX):
    return np.mean(dataX,axis=0)#axis=0表示按照列来求均值，如果输入list,则axis=1
```

3.编写pca方法，具体解释参考注释

```
"""
参数：
	- XMat：传入的是一个numpy的矩阵格式，行表示样本数，列表示特征    
	- k：表示取前k个特征值对应的特征向量
返回值：
	- finalData：参数一指的是返回的低维矩阵，对应于输入参数二
	- reconData：参数二对应的是移动坐标轴后的矩阵
"""
def pca(XMat, k):
    average = meanX(XMat) 
    m, n = np.shape(XMat)
    data_adjust = []
    avgs = np.tile(average, (m, 1))
    data_adjust = XMat - avgs
    covX = np.cov(data_adjust.T)   #计算协方差矩阵
    featValue, featVec=  np.linalg.eig(covX)  #求解协方差矩阵的特征值和特征向量
    index = np.argsort(-featValue) #按照featValue进行从大到小排序
    finalData = []
    if k > n:
        print "k must lower than feature number"
        return
    else:
        #注意特征向量时列向量，而numpy的二维矩阵(数组)a[m][n]中，a[1]表示第1行值
        selectVec = np.matrix(featVec.T[index[:k]]) #所以这里需要进行转置
        finalData = data_adjust * selectVec.T 
        reconData = (finalData * selectVec) + average  
    return finalData, reconData
```


4.编写一个加载数据集的函数

```
#输入文件的每行数据都以\t隔开
def loaddata(datafile):
    return np.array(pd.read_csv(datafile,sep="\t",header=-1)).astype(np.float)
```

5.可视化结果

因为我将维数k指定为2，所以可以使用下面的函数将其绘制出来：

```
def plotBestFit(data1, data2):	  
    dataArr1 = np.array(data1)
    dataArr2 = np.array(data2)
    
    m = np.shape(dataArr1)[0]
    axis_x1 = []
    axis_y1 = []
    axis_x2 = []
    axis_y2 = []
    for i in range(m):
        axis_x1.append(dataArr1[i,0])
        axis_y1.append(dataArr1[i,1])
        axis_x2.append(dataArr2[i,0]) 
        axis_y2.append(dataArr2[i,1])				  
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(axis_x1, axis_y1, s=50, c='red', marker='s')
    ax.scatter(axis_x2, axis_y2, s=50, c='blue')
    plt.xlabel('x1'); plt.ylabel('x2');
    plt.savefig("outfile.png")
    plt.show()	
```

6.测试方法

测试方法写入main函数中，然后直接执行main方法即可：

data.txt可到github中下载：[data.txt](https://github.com/csuldw/MachineLearning/tree/master/PCA/data.txt)

```
#根据数据集data.txt
def main():    
    datafile = "data.txt"
    XMat = loaddata(datafile)
    k = 2
    return pca(XMat, k)
if __name__ == "__main__":
    finalData, reconMat = main()
    plotBestFit(finalData, reconMat)
```

## 结果展示

最后的结果图如下：

<center>
![](./images/output.png)
</center>


蓝色部分为重构后的原始数据，红色则是提取后的二维特征！