## 介绍


在学习机器学习的时候，首当其冲的就是准备一份通用的数据集，方便与其他的算法进行比较。在这里，我写了一个用于加载MNIST数据集的方法，并将其进行封装，主要用于将MNIST数据集转换成numpy.array()格式的训练数据。直接下面看下面的代码吧！

文章链接:[机器学习数据集-MNIST](http://csuldw.github.io/2016/02/25/2016-02-25-machine-learning-MNIST-dataset/)

MNIST数据集原网址：[http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

## 文件目录

- ../../utils/data_util.py 用于加载MNIST数据集方法文件
- ../../utils/test.py 用于测试的文件，一个简单的KNN测试MNIST数据集
- ./train-images.idx3-ubyte  训练集X
- ./train-labels.idx1-ubyte  训练集y
- ./t10k-images.idx3-ubyte   测试集X
- ./t10k-labels.idx1-ubyte   测试集y



## 源码

[../../utils/data_util.py](../../utils/data_util.py)文件 


[../../utils/test.py](../../utils/test.py)文件:简单地测试了一下KNN算法，代码如下

```
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 16:09:58 2016
Test MNIST dataset 
@author: liudiwei
"""

from sklearn import neighbors  
from data_util import DataUtils
import datetime  


def main():
    trainfile_X = '../dataset/MNIST/train-images.idx3-ubyte'
    trainfile_y = '../dataset/MNIST/train-labels.idx1-ubyte'
    testfile_X = '../dataset/MNIST/t10k-images.idx3-ubyte'
    testfile_y = '../dataset/MNIST/t10k-labels.idx1-ubyte'
    train_X = DataUtils(filename=trainfile_X).getImage()
    train_y = DataUtils(filename=trainfile_y).getLabel()
    test_X = DataUtils(testfile_X).getImage()
    test_y = DataUtils(testfile_y).getLabel()

    return train_X, train_y, test_X, test_y 


def testKNN():
    train_X, train_y, test_X, test_y = main()
    startTime = datetime.datetime.now()
    knn = neighbors.KNeighborsClassifier(n_neighbors=3)  
    knn.fit(train_X, train_y)  
    match = 0;  
    for i in xrange(len(test_y)):  
        predictLabel = knn.predict(test_X[i])[0]  
        if(predictLabel==test_y[i]):  
            match += 1  
      
    endTime = datetime.datetime.now()  
    print 'use time: '+str(endTime-startTime)  
    print 'error rate: '+ str(1-(match*1.0/len(test_y)))  

if __name__ == "__main__":
    testKNN()
```

通过main方法，最后直接返回numpy.array()格式的数据：train_X, train_y, test_X, test_y。如果你需要，直接条用main方法即可！

---