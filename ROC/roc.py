# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Created on Tue Aug 11 21:05:25 2015

@author: liudiwei
"""
import numpy as np
import matplotlib.pyplot as plt


def plotROC(predStrengths, classLabels):
    cur = (1.0,1.0) #cursor
    ySum = 0.0 #variable to calculate AUC
    numPosClas = sum(np.array(classLabels)==1.0)
    yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort()#get sorted index, it's reverse
    fig = plt.figure() 
    fig.clf()
    ax = plt.subplot(111)
    #loop through all the values, drawing a line segment at each point
    #for index in sortedIndicies.tolist()[0]:
    for index in sortedIndicies:
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep;
        else:
            delX = xStep; delY = 0;
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve')
    ax.axis([0,1,0,1])
    print "the Area Under the Curve is: ",ySum*xStep 

def loadProbOrClass(file):
    outdata = []
    with open(file, 'r') as fr_pred:
        for eachline in fr_pred:
            outdata.append(float(eachline.strip()))
    return outdata

if __name__=="__main__":
    predStrengths = loadProbOrClass("data/cv_ada.score")
    classLabels = loadProbOrClass("data/label.data")
    plotROC(np.array(predStrengths) , classLabels)
    plt.show()