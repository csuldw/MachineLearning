## Naive Bayes (Python)

说明：本节不适用额外的数据，数据通过loadDataSet构造。

- bayes.py为朴素贝叶斯的实现   

- 库：numpy


- 参考博文：http://www.csuldw.com/2015/05/28/2015-05-28-NB/


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
    testEntry = [['love', 'my', 'dalmation'],
                 ['stupid', 'garbage'],
                 ['Haha', 'I', "Love", "You"],
                 ['This', 'is', "my", "dog"]]
    for item in testEntry:
        clf.testingNB(item)
```


## Output

```
['love', 'my', 'dalmation'] classified as:  0
['stupid', 'garbage'] classified as:  1
['Haha', 'I', 'Love', 'You'] classified as:  0
['This', 'is', 'my', 'dog'] classified as:  0
```

## References

- 机器学习实战
