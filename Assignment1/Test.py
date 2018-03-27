from sklearn.datasets.base import load_iris
from sklearn.cross_validation import train_test_split
from Assignment1.Knn import knn_classifiter
import numpy as np 
from matplotlib import pyplot as plt
from sklearn.neighbors.unsupervised import NearestNeighbors
from sklearn.neighbors.classification import KNeighborsClassifier


print("a")
iris = load_iris()

plt.title("KNN_Accuracy")
plt.xlabel("value of K")
plt.ylabel("Accuracy")
y = []

for i in range(1,150):
    
    X_train,X_test,Y_train,Y_test = train_test_split(iris.data,iris.target,test_size = 0.1)
    kc = knn_classifiter()
    #kc = KNeighborsClassifier()
    kc.fit(X_train, Y_train)
    
    
    result = kc.predict(X_test)
    correct = 0

    for i,j in enumerate(result):
        if(Y_test[i] == j):
            correct += 1

    accuracy = correct / len(Y_test)
    y.append(accuracy)

plt.plot(np.arange(1,150),y)
plt.show()

        