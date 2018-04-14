import numpy as np

class knn_classifiter:
    def __init__(self):
        pass
    
    def fit(self,train_set,labels):
        self.training = train_set
        self.labels = labels
        
    def predict(self,features,k = 3):
        
        y_pred = np.array(np.zeros(features.shape[0])) 
        
        feat2 = np.sum(np.multiply(features,features),axis = 1,keepdims = True).reshape(features.shape[0],1)
        test_feat2 = np.sum(np.multiply(self.training,self.training),axis = 1,keepdims = True).reshape(1,self.training.shape[0])
        dists = feat2 + test_feat2
        dists -= 2 * np.dot(features,self.training.T)
        
        #dists = np.sqrt(dists)
        
        for i in range(features.shape[0]):
            closet = self.labels[np.argsort(dists[i])[:,k]]
            y_pred[i] = np.argmax(np.bincount(closet))
            
        return y_pred
    
    