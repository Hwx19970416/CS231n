'''
Created on 2018年3月26日

@author: SkyNet
'''
import numpy as np

class Softmax:
    def __init__(self,width,k):
        self.weight = np.zeros((k,width),dtype = np.float64)
        self.K = k
        
    def __count_loss(self,prob,y,lamb):
        prob_loss = -np.log(prob)
        total_loss = np.sum(prob_loss[np.arange(prob_loss.shape[0]),y]) / len(y)
        total_loss += (lamb/2) * np.sum(self.weight * self.weight)
        
        return total_loss
    
    
    def train(self,X,y,lamb = 0.000005,learn_rate = 0.0005,training_r = 2000):
        m = X.shape[0]
        n = 0
        loss_recorder = []
        
        while n < training_r:
            
            exp_pri = np.exp(np.dot(X,self.weight.T)).reshape(60000,10)
            #print(exp_pri)
            prob = exp_pri / (np.sum(exp_pri,axis = 1).reshape(60000,1))
            loss = self.__count_loss(prob, y, lamb)  
            
            print("Iteration %d : %f"%(n,loss))
            loss_recorder.append(loss)
            
            prob[np.arange(m),y] = prob[np.arange(m),y] - 1
            
            dw = prob.T.dot(X) / m 
            dw += lamb * self.weight
            self.weight = self.weight - learn_rate * dw
                
            n = n + 1
            
        return loss_recorder
                
    def predict(self,X):
        
        sample = np.asarray(X, dtype = float)
        scores = np.dot(sample,self.weight.T)
        exp_s = np.exp(scores)
        prob_s = exp_s  / np.sum(exp_s)
        result = np.argmax(prob_s)
        
        return result                 