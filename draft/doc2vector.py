# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 14:20:16 2019

@author: liudiwei
"""

# -*- coding: utf-8 -*-
import codecs
import re
from os import listdir

import gensim
import jieba
import numpy as np
import pandas as pd
import os

def doc_segment(doc_path, corpus_path):
    """save word segment
    """
    # 先把所有文档的路径存进一个 array 中，docLabels：
    doc_lists = [file for file in listdir(doc_path) if file.endswith('.txt')]

    for doc in doc_lists:
        try:
            ws = codecs.open(doc_path + "/" + doc, encoding="utf8").read()
            doc_words = segment(ws)
            if not os.path.exists(corpus_path):
                os.mkdir(corpus_path)
            with codecs.open(corpus_path + "/{}".format(doc), "a", encoding="UTF-8") as f:
                f.write(" ".join(doc_words))
        except:
            print(doc)

def segment(doc: str, stopword_file=None):
    """中文分词
    parameter:
        doc : str, input text
    return:
        [type] --- [description]
    """
    # 停用词
    if stopword_file != None:
        stop_words = pd.read_csv(stopword_file, 
                                 index_col=False, 
                                 quoting=3,
                                 names=['stopword'],
                                 sep="\n",
                                 encoding='utf-8')
        
        stop_words = list(stop_words.stopword)
    else:
        stop_words = []
    reg_html = re.compile(r'<[^>]+>', re.S)
    doc = reg_html.sub('', doc)
    doc = re.sub('[0-9]', '', doc)
    doc = re.sub('\s', '', doc)
    word_list = list(jieba.cut(doc))
    out_str = ''
    for word in word_list:
        if word not in stop_words:
            out_str += word
            out_str += ' '
    segments = out_str.split(sep=" ")
    return segments

def build_corpus(corpus_path):
    """build word corpus: list of list
    """
    doc_labels = [f for f in os.listdir(corpus_path) if f.endswith('.txt')]

    corpus = []
    for doc in doc_labels:
        ws = open(corpus_path + "/" + doc, 'r', encoding='UTF-8').read()
        corpus.append(ws)

    print("corpus size: ", len(corpus))
    return corpus, doc_labels

############################## build model ####################################

def train_model(corpus, doc_labels, model_path, model_name="doc2vec.model"):    
    """training model
    parameter:
        - courpus: [[...], [....]]
        - doc_labels: [...]
        - model_path
        - model_name: default value "doc2vec.model"
    return:
        - model: model
        - model_file: model_path + "/" + model_name
    """
    # training doc2vec model and save model to local disk：
    sentences = LabeledLineSentence(corpus, doc_labels)
    # an empty model
    model = gensim.models.Doc2Vec(vector_size=256, 
                                  window=10, 
                                  min_count=5,
                                  workers=4, 
                                  alpha=0.025, 
                                  min_alpha=0.025, 
                                  epochs=12)
    model.build_vocab(sentences)
    
    print("start training...")
    model.train(sentences, total_examples = model.corpus_count, epochs=12)
    
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_file = model_path + "/" + model_name
    model.save(model_file)
    print("Model saved")
    return model, model_file

def test_model(model_file, file1, file2):
    print("Loading Model.")
    model = gensim.models.Doc2Vec.load(model_file)

    sentence1 = open(file1, 'r', encoding='UTF-8').read()
    sentence2 = open(file2, 'r', encoding='UTF-8').read()
    
    # 分词
    print("start to segment")
    words1 = segment(sentence1)
    words2 = segment(sentence2)
    
    # 转成句子向量
    vector1 = sent2vec(model, words1)
    vector2 = sent2vec(model, words2)

    import sys
    print(sys.getsizeof(vector1))
    print(sys.getsizeof(vector2))

    cos = similarity(vector1, vector2)
    print("相似度：{:.4f}".format(cos))


def similarity(a_vect, b_vect):
    """计算两个向量余弦值
    parameter:
        a_vect {[type]} -- a 向量
        b_vect {[type]} -- b 向量
    
    return:
        [type] -- [description]
    """

    dot_val = 0.0
    a_norm = 0.0
    b_norm = 0.0
    cos = None
    for a, b in zip(a_vect, b_vect):
        dot_val += a*b
        a_norm += a**2
        b_norm += b**2
    if a_norm == 0.0 or b_norm == 0.0:
        cos = -1
    else:
        cos = dot_val / ((a_norm*b_norm)**0.5)

    return cos


def sent2vec(model, words):
    """sentence2vector
    parameter:
        model {[type]} -- Doc2Vec 模型
        words {[type]} -- 分词后的文本
    return:
        [type] -- 向量数组
    """
    vect_list = []
    for w in words:
        try:
            vect_list.append(model.wv[w])
        except:
            continue
    vect_list = np.array(vect_list)
    vect = vect_list.sum(axis=0)
    return vect / np.sqrt((vect ** 2).sum())



class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
       self.labels_list = labels_list
       self.doc_list = doc_list

    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield gensim.models.doc2vec.TaggedDocument(words=doc.split(), tags=[self.labels_list[idx]])


if __name__ == '__main__':
    doc_path = "./data/"
    corpus_path = "data/corpus_words"
    model_path = "./models"
    #doc_segment(data_dir)
    corpus, doc_labels = build_corpus(corpus_path)
    model, model_file = train_model(corpus, doc_labels, model_path)
    test_model(model_file, './data/corpus_test/t2.txt', './data/corpus_test/t1.txt')
