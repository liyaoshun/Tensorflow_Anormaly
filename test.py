# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(lys)s
"""

'''
#from keras.models import Sequential
#from keras.layers.core import Dense, Dropout, Flatten
#from keras.layers.convolutional import Deconvolution3D,Conv3D,Convolution3D,UpSampling2D,Convolution2D,MaxPooling2D, MaxPooling3D, ZeroPadding3D
#from keras.optimizers import SGD
#from keras.models import Model
#from keras.layers import Input
#import os
#from keras.models import model_from_json
#import numpy as np
#
#input_img = Input(shape=(28, 28, 1))
#
#x = Convolution2D(32, (3, 3), activation='relu', padding='same')(input_img)
#x = MaxPooling2D((2, 2), border_mode='same')(x)
#x = Convolution2D(32, (3, 3), activation='relu', padding='same')(x)
#encoded = MaxPooling2D((2, 2), border_mode='same')(x)
#
## at this point the representation is (32, 7, 7)
#x = Convolution2D(32, (3, 3), activation='relu', padding='same')(encoded)
#x = UpSampling2D((2, 2))(x)
#x = Convolution2D(32, (3, 3), activation='relu', padding='same')(x)
#x = UpSampling2D((2, 2))(x)
#decoded = Convolution2D(1, (3, 3), activation='sigmoid', padding='same')(x)
#
#autoencoder = Model(input_img, decoded)
#
#print(autoencoder.summary())
'''

#import matplotlib.pyplot as plt
#from PIL import Image,ImageDraw
import cv2
try:
    import xml.etree.cElementTree as ET  #解析xml的c语言版的模块
except ImportError:
    import xml.etree.ElementTree as ET
    
def GetAnnotBoxLoc(AnotPath):#AnotPath VOC标注文件路径
    tree = ET.ElementTree(file=AnotPath)  #打开文件，解析成一棵树型结构
    root = tree.getroot()#获取树型结构的根
    ObjectSet=root.findall('object')#找到文件中所有含有object关键字的地方，这些地方含有标注目标
    ObjBndBoxSet={} #以目标类别为关键字，目标框为值组成的字典结构
    for Object in ObjectSet:
        ObjName=Object.find('name').text
        BndBox=Object.find('bndbox')
        x1 = int(BndBox.find('xmin').text)#-1 #-1是因为程序是按0作为起始位置的
        y1 = int(BndBox.find('ymin').text)#-1
        x2 = int(BndBox.find('xmax').text)#-1
        y2 = int(BndBox.find('ymax').text)#-1
        BndBoxLoc=[x1,y1,x2,y2]
        if ObjBndBoxSet.has_key(ObjName):
        	ObjBndBoxSet[ObjName].append(BndBoxLoc)#如果字典结构中含有这个类别了，那么这个目标框要追加到其值的末尾
        else:
        	ObjBndBoxSet[ObjName]=[BndBoxLoc]#如果字典结构中没有这个类别，那么这个目标框就直接赋值给其值吧
    return ObjBndBoxSet
def draw_box():
    path = "/home/gzs/Documents/denys/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test002/152.tif"
    img = cv2.imread(path,1)
    cv2.rectangle(img,(161,161),(207,207),(0,0,255),1)
    cv2.imshow("bbox ",img)
    cv2.waitKey()
import numpy as np
import matplotlib.pyplot as plt
def draw_hot():
    path="/media/gzs/denys/paper/maha_datas/MasPicMOur/msjpg2/152msj.npy"
#    print(np.load(path).shape)
    data = np.array(np.load(path),np.float32)
    data = np.max(data.reshape((2,30,45)),axis=0, keepdims=True).reshape(30,45)
#    print(data.shape)
#    exit()
    plt.figure(1)
    plt.imshow(data)
    plt.show()
    
if __name__ == "__main__":
#    xml_path = "/home/gzs/Documents/denys/C3D_Anormaly_keras/paper/label/UCSDped2/test1/096.xml"
#    img_path = "/home/gzs/Documents/denys/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test001/096.tif"
#    anno = GetAnnotBoxLoc(xml_path)
#    print(anno['bicycle'])
#    im = cv2.imread(img_path)
#    xmin = anno['bicycle'][0][0]
#    ymin = anno['bicycle'][0][1]
#    xmax = anno['bicycle'][0][2]
#    ymax = anno['bicycle'][0][3]
#    cv2.rectangle(im,(xmin,ymin),(xmax,ymax),(0,0,255),1)# 猩红
#    cv2.line(im,(xmin,ymin),(xmax,ymax),255,1)
#    cv2.imshow('image',im)
#    cv2.waitKey()
#    print(anno)
#    draw_box()
#    draw_hot()
#    xmin = 10
#    ymin = 50
#    minl = np.maximum(xmin,ymin)
#    print(minl)
#    [x1, y1, x2, y2] 
#    其中 (x1, y1) 为左下角的坐标，(x2, y2) 是右上角的坐标
    
    
#    rec2 = [0,0,2,2]
#    rec1 = [1,1,3,3] 
#    rec1 = [0,0,1,1]
#    rec2 = [1,0,2,1] 
#    arr = []
#    arr.append(rec1)
#    arr.append(rec2)
#    print(type(arr))
#    arr = np.array(arr)
#    length = len(arr)
#    print(arr)
#    for i in range(0,length):
#        if (np.array(arr[i])==rec1).all():
#            print("rect1存在")
#            arr = np.delete(arr,i,0)
#            break
#        else:
#            print("rect1不存在")
#    print(type(arr))
    
#    arr = arr.tolist()
#    print(type(arr))
#    print(arr)
#    arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
#    print(arr)
#    arr = np.delete(arr, 1, 0)
#    print(arr)
#    if np.maximum(rec1[0],rec2[0])<np.minimum(rec1[2],rec2[2]) and np.maximum(rec1[1],rec2[1])<np.minimum(rec1[3],rec2[3]):
#        print("true")
#    else:
#        print("false")
#    for i in range(0,10):
#        break_flag = False
#        for j in range(0,10):
#            if j > 5:
#                break_flag = True
#                break
#            else:
#                print("i {0},j {1}".format(i,j))
#        if break_flag:
#            break
    path = "/home/gzs/Documents/denys/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/Train001/120.tif"
    img = cv2.imread(path,1)
#    height,width = img.shape[:2]
    size = (360, 240)  
    shrink = cv2.resize(img, size, interpolation=cv2.INTER_AREA)  
    print("img shape : {0} ".format(shrink.shape))
    cv2.imshow("120.tif",shrink)
    cv2.waitKey()