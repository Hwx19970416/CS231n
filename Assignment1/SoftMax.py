'''
Created on 2018年3月26日

@author: SkyNet
'''
import numpy as np

class Softmax:
    def __init__(self,width,k):
        self.weight = np.random.randn(k,width)
        self.K = k
        
    def __count_loss(self,prob,y,lamb):
        prob_loss = -np.log(prob)
        total_loss = np.sum(prob_loss[prob_loss.shape[0],y])
        total_loss += lamb * np.sum(self.weight * self.weight)
        
        return total_loss
    
    
    def train(self,X,y,lamb = 0.1,learn_rate = 0.01,training_r = 2000):
        m = X.shape[0]
        
        exp_pri = np.exp(X * self.weight.T)
        prob = exp_pri / np.sum(exp_pri,axis = 1)
        
        n = 0
        
        while n < training_r:
            
            loss = self.__count_loss(prob, y, lamb)  
            print(loss)
            prob = prob[m,y] - 1
            
            self.weight -= learn_rate * (X * prob[m,y] + lamb * self.weight)
                 
            n = n + 1
                
    def predict(self,X):
        
        sample = np.asarray(X, dtype = float)
        prob_s = np.dot(self.weight,sample.T)
        prob_s = np.exp(prob_s)
        prob_s /= np.sum(prob_s,axis = 1)
        result = np.argmax(prob_s)
        
        return result                 