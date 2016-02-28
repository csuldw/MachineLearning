# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 19:38:58 2016
similarity
@author: liudiwei
"""

import numpy as np

#欧几里得距离，使用1/(1+distance)归一化
def eulisSim(matA, matB):
    return 1.0/(1.0 + np.linalg.norm(matA - matB))

#pearson相关系数，使用0.5+0.5*corrcof()将数值归一化
def pearsSim(matA, matB):
    if len(matA) < 3:
        return 1.0
    return 0.5 + 0.5 * np.corrcoef(matA, matB, rowvar = 0)[0][1]

#余弦相似度，使用0.5+0.5*cos(theta)归一化
def cosSim(matA, matB):
    val1 = float(matA * matB.T)
    val2 = np.linalg.norm(matA) * np.linalg.norm(matB)
    cos_theta = val1/val2
    return 0.5 + 0.5 * cos_theta
    
if __name__=="__main__":
    matA = np.mat([1,2,3,4,5])
    matB = np.mat([2,3,4,5,6])
    edist = eulisSim(matA, matB)  #output: 0.3090169943749474
    pdist = pearsSim(matA, matB)  #output: 1.0
    cdist = cosSim(matA, matB)    #output: 0.99746833816309111
