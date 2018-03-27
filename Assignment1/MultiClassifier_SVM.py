'''
Created on 2018年3月26日

@author: SkyNet
'''
import numpy as np

class MC_SVM:
    def __init__(self):
        pass
    
    def fit(self,traing,labels,learning_rate):
        self.weight = np.zeros(traing.shape)
        scores = np.dot(traing,self.weight)
        
        margin = scores - scores[np.arange(traing.shape[0]),labels].reshape(traing.shape[0],1)
        margin = (margin > 0) * margin
        
        loss = np.sum() / traing.shape[0]
        loss += 0.5 * learning_rate * np.sum(self.weight * self.weight)
        
        