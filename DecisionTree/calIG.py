# -*- coding: utf-8 -*-
from math import log
import numpy as np

#label 格式为[1, 1 ,..., 0]
def calEntropy(label):
    numEntries = len(label)
    labelCounts = {}
    for labVec in label: #the the number of unique elements and their occurance
        currentLabel = labVec
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) #log base 2
    return shannonEnt


def calIG(dataSet,label):
    baseEntropy=calEntropy(label)#计算无特征的信息熵
    #print  baseEntropy
    num_row=dataSet.shape[0] #特征矩阵的行数
    num_col=dataSet.shape[1] #特征矩阵的列数
    IG={}
    for i in range(num_col): #遍历特征矩阵
        entropy_fea=0        #当前特征的信息熵
        list_fea = dataSet.transpose()[i]  #将特征矩阵转置以获得第i列特征
        class_fea = {}   #第i列特征的类别字典
        num_pos_class={} #第i列特征类别对应标签为正样本的数目字典
        num_neg_class={} #第i列特征类别对应标签为负样本的数目字典
        prob_pos = {}
        prob_neg={}
        num_class=0 #第i列特征的类别数目
        p=0
        for j in range(num_row):#遍历每一列特征的每一行
            cur_fea = list_fea[0,j] #第j行特征
            label_fea= label[j]   #第j行特征对应的标签 
            if cur_fea not in class_fea.keys():class_fea[cur_fea]=None; num_pos_class[cur_fea]= 0;num_neg_class[cur_fea]=0
            #判断第j行特征的类别是否在当前列特征的类别字典内，若不存在添加入类别字典，并对该类别正负样本标签数目初始化为0
            if label_fea == 1:#第j行特征对应的标签是否为正样本
                num_pos_class[cur_fea] += 1
            else:
                num_neg_class[cur_fea] += 1
                
        for cur_class_fea in class_fea:#遍历第i列特征的类别
            num_class=num_pos_class[cur_class_fea]+num_neg_class[cur_class_fea]
            prob_pos[cur_class_fea] = float(num_pos_class[cur_class_fea])/num_class
            #第i列(每一个类别的正样本数目)/(该类别的数量)字典
            prob_neg[cur_class_fea] = float(num_neg_class[cur_class_fea])/num_class
            #第i列(每一个类别的负样本数目)/(该类别的数量)字典
            ratio_pos_class = prob_pos[cur_class_fea]
            #第i列一类别正样本频率    
            ratio_neg_class = prob_neg[cur_class_fea]
            #第i列一类别负样本样本频率
            entropy_pos = 0
            entropy_neg = 0
            if ratio_pos_class!=0:#避免出现0
                entropy_pos=-(ratio_pos_class * log(ratio_pos_class,2))
            if ratio_neg_class!=0:#避免出现0
                entropy_neg=-(ratio_neg_class * log(ratio_neg_class,2))   
            p = float(num_class)/num_row
            #计算每一个类别的信息熵系数
            entropy_fea += p*(entropy_pos+entropy_neg)
            #得第i列特征的信息熵
        IG_fea = baseEntropy - entropy_fea
        #求第i列特征的信息增益 
        IG[i]=IG_fea
        #将第i列特征的信息增益 保存到字典
    return IG

     
def createDataSet():
    dataSet = [[1, 1 ,1 ],
               [0, 1 ,0 ],
               [1, 0 ,1 ]]
               
    labels = [ 1,1,0 ]
    #change to discrete values
    return dataSet, labels


    
dataSet,labels = createDataSet() 

dataMatrix = np.mat(dataSet)


labelMat = labels

print dataMatrix
print labels

IG = calIG(dataMatrix, labelMat)

print IG

    