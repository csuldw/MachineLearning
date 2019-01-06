# -*- coding: utf-8 -*-
"""
Created on Wed Apr 04 09:33:57 2018

@author: liudiwei
"""

import numpy as np
import time

from activators import SigmoidActivator

class FullConnectedLayer(object):
    """
    定义全连接层
    """
    def __init__(self, input_size, output_size, activator):
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        
        # 权重向量W
        self.W = np.random.uniform(-0.1, 0.1, (output_size, input_size))
        
        # 偏置项b
        self.b = np.zeros(output_size)
        # 输出向量
        self.output = np.zeros(output_size)
        
        #内部变量
        self._input = None    
        self._delta = None
        self._W_grad = None
        self._b_grad = None
        
    def _network(self, input_sample, weight, bias):
        return np.dot(weight, input_sample)  + bias
    
    def forward(self, input_sample):
        #前向传播计算输出
        self._input = input_sample
        self.output = self.activator.forward(
                    self._network(self._input, self.W, self.b))
        
    def backward(self, delta_array):
        #后向传播计算权值和偏置  计算前一层delta
        self._delta = self.activator.backward(self._input) * np.dot(
            self.W.T, delta_array)
        
        #传入的数据有点不一样，因此需要加入.reshape(-1,1)
        self._W_grad = np.dot(delta_array.reshape(-1,1), self._input.reshape(-1,1).T)
        self._b_grad = delta_array
        
    def update(self, learning_rate):
        #更新权重值
        self.W += learning_rate * self._W_grad
        self.b += learning_rate * self._b_grad
        
        
class Network(object):
    """
    layers: 一个二维数组，表示神经网络每层的节点数
    """
    def __init__(self, layers):
        
        self.layers = []
        
        for i in range(len(layers) - 1):
            self.layers.append(
                FullConnectedLayer(
                    layers[i], 
                    layers[i+1],
                    SigmoidActivator()
                )
            )
            
    def predict(self, sample):
        #预测某个样本
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def train(self, labels, train_sample, learning_rate, epoch):
        '''
        labels: 样本标签
        train_sample: 输入样本
        learning_rate: 学习速率
        epoch: 训练轮数
        '''
        for i in range(epoch):
            print '%s epoch %d begin.' % (time.ctime(), i+1)
            for j in range(len(train_sample)):
                self._train_one_sample(labels[j], train_sample[j], learning_rate)
            print '%s epoch %d finished' % (time.ctime(), i+1)
    
    
    def _train_one_sample(self, label, sample, learning_rate):
        self.predict(sample)
        self._get_gradient(label)
        self._update_weight(learning_rate)
    
    
    def _get_gradient(self, label):
        #计算输出层delta
        delta = self.layers[-1].activator.backward(self.layers[-1].output) * (
             label - self.layers[-1].output)
        
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer._delta
        return delta
        
    def _update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)



