# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 15:03:51 2019

@author: liudiwei
"""

import os
import pandas as pd
import numpy as np 
import keras
import random

from keras.preprocessing import sequence
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout,Activation
from keras.models import model_from_yaml
from keras.layers import Input, Embedding, Flatten, Dot, Dense, Reshape
from keras.models import Model
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.manifold import TSNE
random.seed(100)

os.chdir(r"../data")


dataset = pd.read_csv('item_rating.csv')

users = list(set(dataset.user_id.tolist()))
items = list(set(dataset.item_id.tolist()))

#获取item的下标和item_id关系
item_index = {item: idx for idx, item in enumerate(items)}
index_item = {idx: item for item, idx in item_index.items()}

#获取user的下标和user_id关系
user_index = {user: idx for idx, user in enumerate(users)}
index_user = {idx: user for user, idx in user_index.items()}

dataset["user_index"] = dataset["user_id"].apply(lambda x: user_index[x])
dataset["item_index"] = dataset["item_id"].apply(lambda x: item_index[x])

def load_data():
    train, test = train_test_split(dataset, test_size=0.2, random_state=42)
    n_users = len(dataset.user_id.unique())
    n_items = len(dataset.item_id.unique())
    print(dataset.head())
    return dataset, train, test, n_users, n_items

dataset, train, test, n_users, n_items = load_data()


def build_model(n_users, n_items, embedding_size = 10, classification=False):
    # Both inputs are 1-dimensional
    user_input = Input(shape=[1], name="user")
    item_input = Input(shape=[1], name="item")

    user_embedding = Embedding(n_users, embedding_size, name="user-Embedding")(user_input)
    item_embedding = Embedding(n_items, embedding_size, name="item-Embedding")(item_input)

    user_vec = Flatten(name="Flatten-Users")(user_embedding)
    item_vec = Flatten(name="Flatten-items")(item_embedding)

    prod = Dot(name="Dot-Product", axes=1)([item_vec, user_vec])
    # Reshape to be a single number (shape will be (None, 1))
    merged = Reshape(target_shape = [1])(prod)

    if classification:
        out = Dense(1, activation = 'sigmoid')(merged)
        model = Model(inputs = [user_input, item_input], outputs = out)
        model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # Otherwise loss function is mean squared error
    else:
        model = Model(inputs = [user_input, item_input], outputs = merged)
        model.compile(optimizer = 'Adam', loss = 'mse')
        #model.compile('adam', 'mean_squared_error')

    model.summary()
    return model

model = build_model(n_users, n_items, embedding_size=20)

def generate_batch(train, 
                   n_positive = 50,
                   negative_ratio = 1.0):
    """Generate batches of samples for training. 
       Random select samples."""
    # Create empty array to hold batch
    batch_size = n_positive *  negative_ratio
    batch = np.zeros((batch_size, 3))
    print(batch_size, batch)
    # Continue to yield samples
    while True:
        # Randomly choose positive examples
        batch_train = train.sample(batch_size)
        batch_train["idx"] = [i for i in range(batch_train.shape[0])]
        def getBatch(x):
            batch[x["idx"], :] = (x["user_index"], x["item_index"], int(x["rating"]))
        batch_train.apply(lambda x: getBatch(x), axis=1)
        
        
        # Make sure to shuffle order
        np.random.shuffle(batch)
        del batch_train
        yield {'user': batch[:, 0], 'item': batch[:, 1]}, batch[:, 2]


#train a regression model with fit
def train_model(model, train, epochs=5):
    model.fit([train.user_index, train.item_index], train.rating, epochs=5, verbose=1)
    model.save('regression_model.h5')

    # Extract embeddings
    item_em = model.get_layer('item-Embedding')
    item_em_weights = item_em.get_weights()[0]
    item_em_weights
    return model, item_em_weights

def train_model_batch(model, train, epochs=5, batch_size_pos=8000):
    n_positive = batch_size_pos
    gen = generate_batch(train, n_positive, negative_ratio = 1)
    # Train
    model.fit_generator(gen, epochs = 5, steps_per_epoch = train.shape[0] // n_positive)
    model.save('regression_model.h5')
    return model

#训练模型Method1
model, item_em_weights = train_model(model, train)
#训练模型Method2
model = train_model_batch(model, train, batch_size_pos=1000)

# Extract item embeddings
item_layer = model.get_layer('item-Embedding')
item_weights = item_layer.get_weights()[0]
item_weights.shape

# Extract user embeddings
user_layer = model.get_layer('user-Embedding')
user_weights = user_layer.get_weights()[0]
user_weights.shape


#PCA visualization
def pca_show(weights):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(weights)
    sns.jointplot(x=pca_result[:,0], y=pca_result[:,1])
pca_show(item_em_weights)



def tsne_show(weights):
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tnse_results = tsne.fit_transform(weights)
    sns.jointplot(x=tnse_results[:,0], y=tnse_results[:,1])
tsne_show(item_em_weights)


#recommendation for the user 
def recommend_for_user(user_index, dataset):
    item_data = np.array(list(set(dataset.item_index)))
    user = np.array([user_index for i in range(len(item_data))])
    predictions = model.predict([user, item_data])
    predictions = np.array([a[0] for a in predictions])
    recommended_item_ids = (-predictions).argsort()[:30]
    print(recommended_item_ids)
    return recommended_item_ids
    #print(predictions[recommended_item_ids])

aaa = recommend_for_user(0,dataset)
aaa = recommend_for_user(3,dataset)
aaa = recommend_for_user(4,dataset)

#calculate cosine similarity 
from sklearn.metrics.pairwise import cosine_similarity
cos = cosine_similarity(item_em_weights[0:1], item_em_weights)
recommendations = cos[0].argsort()[-4:][::-1]

