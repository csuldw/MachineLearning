## Decision Tree

决策树理论详解：http://www.csuldw.com/2015/05/08/2015-05-08-decision%20tree/

- data存放数据集
- calIG.py：计算信息增益的实例代码
- DT.py：决策树实现
- treePlotter.py：决策树的可视化绘制

## 相关知识

- python
- numpy
- matplotlib

## dataset

- 训练集：./data/train.txt
- 测试集：./data/test.txt

## Run

```
if __name__=="__main__":
    trainfile=r"data\train.txt"
    testfile=r"data\test.txt"
    import sys
    sys.path.append(r"F:\CSU\Github\MachineLearning\lib")  
    import dataload as dload
    train_x, train_y = dload.loadData(trainfile)
    test_x, test_y = dload.loadData(testfile)
    
    clf = DecitionTree(criteria="C4.5")
    clf.fit(train_x, train_y)
    result = clf.predict(test_x)    
    outpdf = r"tree.pdf"
    #clf.show(outpdf)
```

## Result

训练得到的树：https://github.com/csuldw/MachineLearning/tree/master/DecisionTree/tree.pdf

对test分类的结果：

```
1-th sample is classfied as: 1
2-th sample is classfied as: 0
3-th sample is classfied as: 0
```

## 参考资料

- 机器学习实战
- Andrew Ng 机器学习公开课