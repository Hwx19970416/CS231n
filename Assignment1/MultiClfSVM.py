'''
Created on 2018年4月14日

@author: SkyNet
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.decomposition.pca import PCA
from sklearn.cross_validation import train_test_split
import time

class MultiSVM:
    def __init__(self,k,feat):
        self.weight = np.random.randn(feat,k)
        pass
    
    def loss_function(self,X,y,reg):
        
        samp_num = X.shape[0]
        dW = np.zeros(self.weight.shape)
        
        score = X.dot(self.weight)
        margin = score - score[np.arange(samp_num),y].reshape(samp_num,1) + 1
        
        margin[np.arange(samp_num),y] = 0
        
        margin = (margin > 0) * margin
        
        loss = margin.sum() / samp_num
        loss += 0.5 * reg * np.sum(self.weight * self.weight)
        
        pass
    
        count = (margin > 0).astype(int)
        count[np.arange(samp_num),y] = -np.sum(count,axis = 1)
        dW += np.dot(X.T,count) / samp_num + reg * self.weight
         
        return loss,dW
    
    def train(self,X,y,learning_rate = 0.1,reg = 0.001,loop = 2000):
        num = X.shape[0]
        n = 0
        
        X = np.asarray(X)
        y = np.asarray(y)
        
        while n < loop:
            loss,dW = self.loss_function(X, y, reg)
            self.weight = self.weight - learning_rate * dW
            
            print("The loss of iteration {num} is {loss}".format(num = n,loss = loss))
            n += 1
            
    def predict(self,X):
        X = np.asarray(X) 
        res_sc = X.dot(self.weight)
        res = np.argmax(res_sc, axis = 1)
        
        #print(type(res))
        
        return res

def show_image(row,sizex = 28,sizeh = 28,labels = True):
    if labels:
        label = row[0]
        pixels = row[1:]
    else:
        label = ""
        pixels = row[0:]
        
    pixels = np.asarray(pixels, dtype = "uint8")
    #print(pixels)
    pixels = pixels.reshape((28,28))
    
    if labels:
        plt.title("Label is {label}".format(label  = label))
    plt.imshow(pixels, cmap = "gray")
        
def plot_slice(rows,sizex = 28,sizeh = 28,labels = True):
    
    num = rows.shape[0]

    w = 4
    h = math.ceil(num / 4)
    
    fig,plot = plt.subplots(h,w)
    fig.tight_layout()
    
    for i in range(num):
        s = plt.subplot(h,w,i+1)
        show_image(rows.ix[i], sizex, sizeh, labels)
        
    plt.show()
    
train = pd.read_csv("D:/kaggleDataSets/MNIST/train.csv")
print(train.shape)
plot_slice(train[0:16])

X_train = train.drop(labels = "label", axis = "columns", inplace = False)
y_train = train["label"]

X_tr,X_ts,y_tr,y_ts = train_test_split(X_train,y_train,test_size = 0.3)
#print(np.asarray(y_ts))
print("PCA algorithm launched")
ts = time.time()
pca = PCA(n_components = 16, whiten = True, svd_solver = "randomized").fit(X_tr)
print("PCA done {time}".format(time = (time.time() - ts) * 1000))

X_tr = pca.transform(X_tr)
X_ts = pca.transform(X_ts)

svc = MultiSVM(10,16)
svc.train(X_tr, y_tr)

res = svc.predict(X_ts)
count = 0

for i in range(0,res.shape[0]):
    if res[i] == np.asarray(y_ts)[i]:
        count += 1

print(count/res.shape[0])
    