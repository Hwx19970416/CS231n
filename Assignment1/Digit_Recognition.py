'''
Created on 2018年3月28日

@author: SkyNet
'''
import os
import struct
import numpy as np
from Assignment1.SoftMax import Softmax
from matplotlib import pyplot as plt


def load_data(path,type = "train"):
    labels_path = os.path.join(path,"%s-labels.idx1-ubyte"%type)
    images_path = os.path.join(path,"%s-images.idx3-ubyte"%type)
    
    with open(labels_path,"rb") as lp:    
        """
        TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
        [offset] [type]          [value]          [description]
        0000     32 bit integer  0x00000801(2049) magic number (MSB first)
        0004     32 bit integer  60000            number of items
        0008     unsigned byte   ??               label
        0009     unsigned byte   ??               label
        ........
        xxxx     unsigned byte   ??               label
        The labels values are 0 to 9.
    
        :param idx_ubyte_file: idx文件路径
        :return: n*1维np.array对象，n为图片数量
        """
        lab_magic,lab_items = struct.unpack(">II",lp.read(8))
        labels = np.fromfile(lp,dtype = np.uint8)
        
    with open(images_path,"rb") as ip:
        """
        TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
        [offset] [type]          [value]          [description]
        0000     32 bit integer  0x00000803(2051) magic number
        0004     32 bit integer  60000            number of images
        0008     32 bit integer  28               number of rows
        0012     32 bit integer  28               number of columns
        0016     unsigned byte   ??               pixel
        0017     unsigned byte   ??               pixel
        ........
        xxxx     unsigned byte   ??               pixel
        Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

        :param idx_ubyte_file: idx文件路径
        :return: n*row*col维np.array对象，n为图片数量
        """
        img_magic,img_items,row,col = struct.unpack(">IIII",ip.read(16))
        images = np.fromfile(ip,dtype = np.uint8).reshape(len(labels),784)
        
    return images,labels
    

images,labels = load_data("E:\image_recognition")

images = np.asarray(images, dtype = float )
images /=255
sm_classifiter = Softmax(784,10)

loss = sm_classifiter.train(images, labels, learn_rate = 0.01)


images,labels = load_data("E:\image_recognition", type = "t10k")

images = np.asarray(images,dtype = float)
images /= 255
leng = len(labels)
correct = 0
for i in range(leng):
    res = sm_classifiter.predict(images[i])
    if res == labels[i]:
        correct += 1
    

print(correct / leng)
plt.title("Loss function")
plt.xlabel("iterations")
plt.ylabel("loss")
plt.plot(range(0,2000),loss)
plt.show()
    
    
    
    