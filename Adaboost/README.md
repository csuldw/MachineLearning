## Adaboost

参考博文：[机器学习算法-Adaboost](http://www.csuldw.com/2015/07/05/2015-07-05-ML-algorithm-Adaboost/)

- Python + Numpy

Adaboost是Ensemble算法中比较经典的一种，其运行过程为：训练集中的每个样本，赋予其一个权重，这些权重构成向量D。一开始，这些权重都初试化成相等值。首先在训练数据上训练处一个若分类器并计算该分类器的错误率，然后在同一数据集上再次训练若分类器。在分类器的第二次训练当中，将会重新调整每个样本的权重，其中第一次分队的样本的权重值将会降低，而第一次分错的样本的权重将会提高。为了从所有分类器中得到最终的分类结果，AdaBoost为每个分类器都分配了一个权重值alpha，这些alpha值是基于每个分类器的错误率进行计算的。计算出alpha值之后，可以对权重向量D进行更新，使得正确分类的样本的权重值降低而分错的样本权重值升高，计算出D后，AdaBoost接着开始下一轮的迭代。AdaBoost算法会不断地重复训练和调整权重的过程，知道训练错误率为0或者若分类器的数目达到用户指定值为止。全文详解请[点击这里](http://www.csuldw.com/2015/07/05/2015-07-05-ML-algorithm-Adaboost/).

## 目录介绍

- data: 存放数据集
- adaboost.py：算法实现，封装成class
- HandWriting.py：用adaboost算法测试手写识别字，用来和KNN进行比较
- test.py：数据集与logistics_regression的数据集一样

## Results

test.py输出error_rate的结果为：[[ 0.29850746]]



