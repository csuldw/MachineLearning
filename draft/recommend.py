# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 15:34:34 2019

@author: liudiwei
"""
import pandas as pd
import numpy as np
import random

from keras.layers import Input, Embedding, Dot, Reshape, Dense
from keras.models import Model
random.seed(100)

#load dataset
user_keywords = pd.read_csv("../data_process/user_keywords.csv")

"""
   user_id                                   keywords
0      113  新闻推荐|资讯推荐|内容推荐|文本分类|人工分类|自然语言处理|聚类|分类|冷启动
1      143                         网络|睡眠|精神衰弱|声音|人工分类
2      123                          新年愿望|梦想|2018|辞旧迎新
3      234                      父母|肩头|饺子|蔬菜块|青春叛逆期|声音
4      117       新闻推荐|内容推荐|文本分类|人工分类|自然语言处理|聚类|分类|冷启动
5      119            新闻推荐|资讯推荐|人工分类|自然语言处理|聚类|分类|冷启动
6       12              新闻推荐|资讯推荐|内容推荐|文本分类|聚类|分类|冷启动
7      122                   机器学习|新闻推荐|梦想|人工分类|自然语言处理
"""

def date_process(user_item):
    """user_item is a DataFrame, column=[user_id, keywords]   
    1. user_item: user and item information, user_id, keywords, keyword_index
    2. user_index: user to index
    3. item_index：item to index
    """
    user_item["keywords"] = user_item["keywords"].apply(lambda x: x.split("|"))
    keyword_list = [] 
    for i in user_item["keywords"]:
        keyword_list.extend(i)
        
    #word count
    item_count = pd.DataFrame(pd.Series(keyword_list).value_counts()) 
    # add index to word_count
    item_count['id'] = list(range(0, len(item_count)))
    
    #将word的id对应起来
    map_index = lambda x: list(item_count['id'][x])
    user_item['keyword_index'] = user_item['keywords'].apply(map_index) #速度太慢
    #create user_index, item_index
    user_index = { v:k for k,v in user_item["user_id"].to_dict().items()}
    item_index = item_count["id"].to_dict()
    return user_item, user_index, item_index

user_keywords, user_index, keyword_index = date_process(user_keywords)

def create_pairs(user_keywords, user_index):
    """
    generate user, keyword pair list
    """
    pairs = []
    def doc2tag(pairs, x):
        for index in x["keyword_index"]:
            pairs.append((user_index[x["user_id"]], index))
    user_keywords.apply(lambda x: doc2tag(pairs, x), axis=1) #速度太慢
    return pairs

pairs = create_pairs(user_keywords, user_index)


def build_embedding_model(embedding_size = 50, classification = False):
    """Model to embed users and keywords using the Keras functional API.
       Trained to discern if a keyword is clicked by user"""
    
    # Both inputs are 1-dimensional
    user = Input(name = 'user', shape = [1])
    keyword = Input(name = 'keyword', shape = [1])
    
    # Embedding the user default: (shape will be (None, 1, 50))
    user_embedding = Embedding(name = 'user_embedding',
                               input_dim = len(user_index),
                               output_dim = embedding_size)(user)
    
    # Embedding the keyword default: (shape will be (None, 1, 50))
    keyword_embedding = Embedding(name = 'keyword_embedding',
                               input_dim = len(keyword_index),
                               output_dim = embedding_size)(keyword)
    
    # Merge the layers with a dot product along the second axis 
    # (shape will be (None, 1, 1))
    merged = Dot(name = 'dot_product', normalize = True,
                 axes = 2)([user_embedding, keyword_embedding])
    
    # Reshape to be a single number (shape will be (None, 1))
    merged = Reshape(target_shape = [1])(merged)
    
    # Squash outputs for classification
    out = Dense(1, activation = 'sigmoid')(merged)
    model = Model(inputs = [user, keyword], outputs = out)
    
    # Compile using specified optimizer and loss 
    model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', 
                  metrics = ['accuracy'])
    #print(model.summary())
    return model

model = build_embedding_model(embedding_size = 20, classification = False)


def generate_batch(pairs, n_positive = 50, negative_ratio = 1.0):
    """Generate batches of samples for training. 
       Random select positive samples
       from pairs and randomly select negatives."""
    
    # Create empty array to hold batch
    batch_size = n_positive * (1 + negative_ratio)
    batch = np.zeros((batch_size, 3))
    
    # Continue to yield samples
    while True:
        # Randomly choose positive examples
        for idx, (user_id, keyword_id) in enumerate(random.sample(pairs, n_positive)):
            batch[idx, :] = (user_id, keyword_id, 1)
        idx += 1
        
        # Add negative examples until reach batch size
        while idx < batch_size:
            
            # Random selection
            random_user = random.randrange(len(user_index))
            random_keyword = random.randrange(len(keyword_index))
            #print(random_user, random_keyword)
            
            # Check to make sure this is not a positive example
            if (random_user, random_keyword) not in pairs:
                
                # Add to batch and increment index
                batch[idx, :] = (random_user, random_keyword, 0)
                idx += 1
                
        # Make sure to shuffle order
        np.random.shuffle(batch)
        yield {'user': batch[:, 0], 'keyword': batch[:, 1]}, batch[:, 2]
        
        
n_positive = len(pairs)
gen = generate_batch(pairs, n_positive, negative_ratio = 1)
# Train
h = model.fit_generator(gen, epochs = 100, steps_per_epoch = len(pairs) // n_positive)


# Extract embeddings
user_layer = model.get_layer('user_embedding')
user_weights = user_layer.get_weights()[0]


keyword_layer = model.get_layer('keyword_embedding')
keyword_weights = keyword_layer.get_weights()[0]

from sklearn.decomposition import PCA
import seaborn as sns

#PCA可视化
def pca_show():
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(user_weights)
    sns.jointplot(x=pca_result[:,0], y=pca_result[:,1])
pca_show()


#calculate cosine similarity 
from sklearn.metrics.pairwise import cosine_similarity
cos = cosine_similarity(user_weights[0:1], user_weights)
recommendations = cos[0].argsort()[-4:][::-1]
