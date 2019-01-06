# -*- coding: utf-8 -*-
"""
Created on Wed Apr 04 17:12:30 2018

@author: liudiwei
"""
import numpy as np
from bp import Network
import pandas as pd

def output_value(labelVec):
    max_value_index = 0
    max_value = 0
    for i in range(len(labelVec)):
        if labelVec[i] > max_value:
            max_value = labelVec[i]
            max_value_index = i
    return max_value_index

def evaluate(network, test_data_set, test_labels):
    error = 0
    total = len(test_data_set)
    for i in range(total):
        label = output_value(test_labels[i])
        predict = output_value(network.predict(test_data_set[i]))
        
        if label != predict:
            error += 1
            print "label is %s" %label
            print "predict value is %s" %predict
            
    return float(error) / float(total)

# one_hot encoder
#error ratio is 0.053167
#error ratio is 0.063917
def onehot_encoder(label, classes):
    label_size = label.shape[0]
    label_onehot = np.zeros((label_size, classes)) + 0.1
    for i in range(label_size):
        label_onehot[i][int(label[i])] = 0.9
    return label_onehot

if __name__ == "__main__":
    
    train_data = pd.read_csv('./data/train.csv')
    testData = pd.read_csv("./data/test.csv")
    
    train_input = train_data.iloc[:, 1:].values.astype(np.float)
    data = np.multiply(train_input, 1.0/255.0)
    label = train_data.iloc[:, 0].values.astype(np.float)
    
    label = onehot_encoder(label, 10)    
    
    train_data = data[:30000]
    train_labels = label[:30000]    
    
    test_data = data[30000:]
    test_labels = label[30000:]
    
    network = Network([784, 300, 10])
    learning_rate = 0.3
    epoch = 5
    network.train(label, data, learning_rate, epoch)
    
    #error_ratio = evaluate(network, test_data, test_labels)
   # print "error ratio is %f" %error_ratio
            
    ###########################################################################        
    test_input = testData.values.astype(np.float)
    test_input = np.multiply(test_input, 1.0/255.0)            
    
    predicts = np.zeros(len(test_input))
    for i in range(len(test_input)):
        predict = output_value(network.predict(test_input[i]))
        predicts[i] = predict
    
    submissions = pd.DataFrame({'ImageId': np.arange(1 , 1 + test_input.shape[0]), 'Label': predicts.astype(int)})
    submissions.to_csv('./submission.csv', index=False)
