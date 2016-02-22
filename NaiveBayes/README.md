## Naive Bayes (Python)

说明：本节不适用额外的数据，数据通过loadDataSet构造。首先通过类别标签计算各个类别各自出现的概率各为多少，即P(Yi)；然后计算每个类别下某个特征取某个数值的概率P(Wi|Yi)；最后将两者相乘。为了解决条件概率相乘时乘积下溢情况，通常采取对每个数取对数然后求和的方式解决。

- bayes.py为朴素贝叶斯的实现：

- 库：numpy


- 参考博文：http://www.csuldw.com/2015/05/28/2015-05-28-NB/

简而言之，朴素贝叶斯是生成式模型的典型例子。关于生成式模型和判别式模型参考：[判别模型 和 生成模型](http://blog.sciencenet.cn/home.php?mod=space&uid=248173&do=blog&id=227964)

常见的判别模型有线性回归、对数回归、线性判别分析、支持向量机、boosting、条件随机场、神经网络等。

常见的生产模型有HMM、朴素贝叶斯模型、高斯混合模型、LDA、Restricted Boltzmann Machine等。

## Test

```
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
```


## Output

```
['love', 'my', 'girl', 'friend'] classified as:  0
['stupid', 'garbage'] classified as:  1
['Haha', 'I', 'really', 'Love', 'You'] classified as:  0
['This', 'is', 'my', 'dog'] classified as:  0
```

## References

- 机器学习实战
