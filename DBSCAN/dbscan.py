#!/usr/bin/env python
# -*- coding: utf-8 -*-

#           实现了DBSCAN算法

__author__ = 'ZYC@BUPT'
import jieba
import os
import sys
import json
jieba.load_userdict("newword.dict")
sys.setdefaultencoding("utf-8")
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
import math
import time
UNCLASSIFIED = False
NOISE = 0

def Test2(rootDir):
    for lists in os.listdir(rootDir):
        path = os.path.join(rootDir, lists)
       # print path.decode('gb2312')
        if path.find(".txt")!=-1:
            Participle(path)
        if os.path.isdir(path):
            Test2(path)

def Participle(path):
    try:
        fp = open(path, "r")
        ad = fp.readline().strip('\n')
        na = fp.readline().strip('\n')
        ti = fp.readline().strip('\n')#time
        si = fp.readline().strip('\n')
        cont = na+fp.read()
        fp.close()
    except IOError:
        return 0

    try:
        insi = {}
        insi['time'] = ti
        print(ti)
        insi['url'] = ad
        insi['title'] = na
        insi['site'] = si#decode("gb2312").encode("utf-8")
        global fnum
        global segcont
        global doc
        seg_list = jieba.lcut(cont, cut_all=False)
        stline = ""
        for word in seg_list:
            if ((word in d) is False) and word != '\n':
                stline = stline + " " + word
        segcont.append(stline)
        print (str(fnum) + " 分词")
        doc[fnum] = insi
        fnum = fnum + 1
    except UnicodeError:
        return 0

def loadDataSet(splitChar=','):
    dataSet = []
    global  we
    dataSet=we
    del we
    return dataSet

def region_query(data, pointId, eps):
    nPoints = data.shape[1]
    seeds = []
    for i in range(nPoints):
        if eps_neighbor(data[:, pointId], data[:, i], eps):
            seeds.append(i)
    return seeds

def tstore(clusters,clusterNum):#测试使用
    global doc
    fpath="./test_res/"
    i=0
    wr=[]
    while i<=clusterNum:
        path=fpath+str(i)+".txt"
        fp=open(path,'w')
        wr.append(fp)
        i+=1
    i=1
    for cl in clusters:
        enstr=""
        enstr=doc[i]['title']+doc[i]['url']
        wr[cl].write(enstr+'\n')
        i+=1

def expand_cluster(data, clusterResult, pointId, clusterId, eps, minPts):
    seeds = region_query(data, pointId, eps)
    if len(seeds) < minPts: # 不满足minPts条件的为噪声点
        clusterResult[pointId] = NOISE
        return False
    else:
        clusterResult[pointId] = clusterId # 划分到该簇
        for seedId in seeds:
            clusterResult[seedId] = clusterId
        while len(seeds) > 0: # 扩张
            currentPoint = seeds[0]
            queryResults = region_query(data, currentPoint, eps)
            if len(queryResults) >= minPts:
                for i in range(len(queryResults)):
                    resultPoint = queryResults[i]
                    if clusterResult[resultPoint] == UNCLASSIFIED:
                        seeds.append(resultPoint)
                        clusterResult[resultPoint] = clusterId
                    elif clusterResult[resultPoint] == NOISE:
                        clusterResult[resultPoint] = clusterId
            seeds = seeds[1:]

        return True

def dbscan(data, eps, minPts):
    clusterId = 1
    nPoints = data.shape[1]
    clusterResult = [UNCLASSIFIED] * nPoints
    for pointId in range(nPoints):
       # print "point :"+str(pointId)
        point = data[:, pointId]
        if clusterResult[pointId] == UNCLASSIFIED:
            if expand_cluster(data, clusterResult, pointId, clusterId, eps, minPts):
                clusterId = clusterId + 1
    return clusterResult, clusterId - 1


def eps_neighbor(a, b, eps):
    dis=math.sqrt(np.power(a - b, 2).sum())
    print(dis)
    return dis < eps

def main():
    dataSet = loadDataSet(splitChar=',')
    dataSet = np.mat(dataSet).transpose()
    # print(dataSet)
    clusters, clusterNum = dbscan(dataSet, 1.37, 5)#################################
    print("cluster Numbers = ", clusterNum)
    # print(clusters)
    #store(clusters, clusterNum)
    tstore(clusters, clusterNum)


def TFIDF():
    global segcont
    global weight
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(segcont))
    word = vectorizer.get_feature_names()  # 所有文本的关键字
    weight = tfidf.toarray()  # 对应的tfidf矩阵
    del segcont

    seg = []
    for i in range(len(weight)):
        enstr = ""
        for j in range(len(word)):
            if weight[i][j] >= 0.1:#####################################
                enstr = enstr + " " + word[j]
        seg.append(enstr)

    del weight
    vec = CountVectorizer()
    tra = TfidfTransformer()
    tidf = tra.fit_transform(vec.fit_transform(seg))
    wo = vec.get_feature_names()
    we = tidf.toarray()

    global we

def dbs():
    global fnum,doc,segcont,d
    fnum = 1
    segcont = []
    doc = {}
    stfp = open("stop.txt", "r")
    stcont = stfp.read()
    list_a = jieba.lcut(stcont, cut_all=False)
    d = set([])
    for li in list_a:
        d.add(li)
    stfp.close()
    Test2('./sou1')
    TFIDF()
    start = time.clock()
    main()
    end = time.clock()
    print('finish all in %s' % str(end - start))

dbs()




