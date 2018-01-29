# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 09:20:04 2018

@author: liudiwei
"""

class Perceptron(object):
    def __init__(self, input_num, activation_func):
        """init parameter
        activation_func: this is a function of activation
        input_num: number of sample
        """
        self.activation_func = activation_func
        self.weights = [0.0 for _ in range(input_num)]
        self.bias = 0.0           
    
    def train(self, input_vecs, labels, iteration, rate):
        """training model
        input_vec: input vetcor, a 2-D list
        labels: class label list
        iteration: 
        rate: learning rate
        """
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)
            
    def _one_iteration(self, input_vecs, labels, rate):
        """training model on input_vecs dataset"""
        samples = zip(input_vecs, labels)
        for (input_vec, class_label) in samples:
            output_val = self.predict(input_vec)
            self._update_weights(input_vec, output_val, class_label, rate)

    def _update_weights(self, input_vec, output_val, class_label, rate):
        """update weights for each iteration"""
        delta = class_label - output_val
        self.weights = map(lambda (x, w): w + rate * delta * x, 
                           zip(input_vec, self.weights))
        self.bias += rate * delta
    
    def __to_string__(self):
        return 'weights\t: %s\nbias\t: %f\n' % (self.weights, self.bias)
        
    def predict(self, input_vec):
        """input input_vec and return a prediction value"""
        return self.activation_func(
            reduce(lambda a, b: a + b,
                    map(lambda (x, w): x * w, zip(input_vec, self.weights)),
                    0.0) + self.bias)
