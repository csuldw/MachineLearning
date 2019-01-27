# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 23:18:00 2019

@author: liudiwei
"""
import numpy as np
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd

user_items = pd.read_csv("./data_process/user_keywords.csv")
user_items["items"] = user_items["keywords"].apply(lambda x: x.replace("|", " "))
corpus = user_items["items"].tolist()

tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)

transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值

vectorizer=CountVectorizer()#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频

tfidf=transformer.fit_transform(tfidf_vectorizer.fit_transform(corpus))#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵

word=tfidf_vectorizer.get_feature_names()#获取词袋模型中的所有词语
weight=tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重


#calculate cosine similarity 
from sklearn.metrics.pairwise import cosine_similarity
cos = cosine_similarity(weight[0:1], weight)
recommendations = cos[0].argsort()[-4:][::-1]


a = list(weight[0:1][0])
aaa = pd.DataFrame(weight)
cosine_similarity(np.array([a]), aaa)