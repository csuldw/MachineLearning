# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 22:07:33 2015

@author: liudiwei
"""

print(__doc__)
import numpy as np
import matplotlib as mpt
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig  
from sklearn.metrics import roc_curve, auc
mpt.use('TKAgg')  

#flag为true时，将其转化为float行，为false时，转为int型
def loadProbOrClass(inputfile, flag):
    outdata = []
    with open(inputfile, 'r') as fr_pred:
        for eachline in fr_pred:
            oneline = eachline.split('\t')
            if flag == True:
                outdata.append(float(oneline[-1].strip()))
            else:
                outdata.append(int(oneline[-1].strip()))
    return outdata

def combineArray(arr1, arr2, arr3, arr4):
    outarr = []
    for i in range(len(arr1)):
        num = []    
        num.append(arr1[i])
        num.append(arr2[i])
        num.append(arr3[i])
        num.append(arr4[i])
        outarr.append(num)
    return outarr

def plotROC(y_score, labels, outpdf):
    n_classes = labels.shape[1]
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.figure(figsize = (6,6))
    
    # Plot ROC curve
    for i in range(4):
        plt.plot(fpr[i], tpr[i], label='' + classifiers[i]+ ' AUC={1:0.2f}'
                                       ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False positive rate(1-Specificity)')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    savefig(outpdf)  
    plt.show()

if __name__=="__main__":    
    pred1 = np.array(loadProbOrClass("data/cv_gbdt.score", True))
    pred2 = np.array(loadProbOrClass("data/cv_svm.score", True))
    pred3 = np.array(loadProbOrClass("data/cv_ada.score", True))
    pred4 = np.array(loadProbOrClass("data/cv_rf.score", True))
    
    classifiers = [ "gbdt", "SVM", "Adaboost", 'RF']
    pred = combineArray(pred1, pred2, pred3, pred4)
    classlabel = np.array(loadProbOrClass("data/label.data", False))
    classLabels = combineArray(classlabel, classlabel, classlabel, classlabel)
    
    
    # y_score:float64; y_test: int32
    y_score = np.array(pred)
    labels = np.array(classLabels)
    outpdf = 'figure_ROC.pdf'
    plotROC(y_score, labels, outpdf)
    