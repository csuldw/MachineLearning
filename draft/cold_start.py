# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 15:03:51 2019

@author: liudiwei
"""

import os
import pandas as pd
import numpy as np 
import keras

from keras.preprocessing import sequence
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout,Activation
from keras.models import model_from_yaml
from keras.layers import Input, Embedding, Flatten, Dot, Dense
from keras.models import Model
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.manifold import TSNE

os.chdir(r"../data")

def load_data():
    dataset = pd.read_csv('ratings.csv')
    train, test = train_test_split(dataset, test_size=0.2, random_state=42)
    n_users = len(dataset.user_id.unique())
    n_books = len(dataset.book_id.unique())
    print(dataset.head())
    return dataset, train, test, n_users, n_books

dataset, train, test, n_users, n_books = load_data()

def build_model(n_users, n_books):
    book_input = Input(shape=[1], name="Book-Input")
    book_embedding = Embedding(n_books+1, 5, name="Book-Embedding")(book_input)
    book_vec = Flatten(name="Flatten-Books")(book_embedding)
    
    user_input = Input(shape=[1], name="User-Input")
    user_embedding = Embedding(n_users+1, 5, name="User-Embedding")(user_input)
    user_vec = Flatten(name="Flatten-Users")(user_embedding)

    prod = Dot(name="Dot-Product", axes=1)([book_vec, user_vec])
    model = Model([user_input, book_input], prod)
    model.compile('adam', 'mean_squared_error')
    
    model.summary()
    return model

model = build_model(n_users, n_books)

#train model 
def train_model(model, train):
    model.fit([train.user_id, train.book_id], train.rating, epochs=1, verbose=1)
    model.save('regression_model.h5')

    # Extract embeddings
    book_em = model.get_layer('Book-Embedding')
    book_em_weights = book_em.get_weights()[0]
    book_em_weights
    return model, book_em_weights

model, book_em_weights = train_model(model, train)


#PCA可视化
def pca_show(weights):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(weights)
    sns.jointplot(x=pca_result[:,0], y=pca_result[:,1])
pca_show(book_em_weights)

def tsne_show(weights):
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tnse_results = tsne.fit_transform(weights)
    sns.jointplot(x=tnse_results[:,0], y=tnse_results[:,1])
tsne_show(book_em_weights)

#recommendation for the first user 
def recommend_for_user(dataset):
    book_data = np.array(list(set(dataset.book_id)))
    user = np.array([1 for i in range(len(book_data))])
    predictions = model.predict([user, book_data])
    predictions = np.array([a[0] for a in predictions])
    recommended_book_ids = (-predictions).argsort()[:30]
    print(recommended_book_ids)
    print(predictions[recommended_book_ids])

recommend_for_user(dataset)