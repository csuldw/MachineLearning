# ROC曲线

- roc.py: 一个用于绘制ROC曲线的函数，重新实现。    
- roc_cv.py  用于在一个图中绘制多个ROC曲线，调用了scikit-learn的计算roc的函数.

# Data

- 位于data目录下，label.data为类别标签，cv_ada.score,cv_gbdt.score,cv_fr.score,cv_svm.score四个文件分别是通过四种分类器训练出来的score值。

#Result

- 最后的结果输出为是figure_ROC.pdf