# -*- coding: utf-8 -*-
"""
Created on 2018 5 30
@function 训练过程中需要使用到的小功能集合
@author: 李谣顺
"""
import cv2,numpy as np
np.set_printoptions(threshold=np.nan)
import os
#import math
#import random
#import time
#import datetime
#from keras.utils import np_utils
#import mahalanobis as msd
from matplotlib import pyplot as plt
#from keras.preprocessing import image
#import h5py as h5
def daw_14(data,pic_path):
    to_array = np.array(data)
    reshape_array = to_array.reshape((14,14))
    plt.figure(1)
    plt.imshow(reshape_array)
    plt.savefig(pic_path,format='png', bbox_inches='tight', transparent=True)
    plt.close()
def daw_22(data,pic_path):
    to_array = np.array(data)
    reshape_array = to_array.reshape((22,22))
    plt.figure(1)
    plt.imshow(reshape_array)
    plt.savefig(pic_path,format='png', bbox_inches='tight', transparent=True)
    plt.close()
def daw_26(data,pic_path):
    to_array = np.array(data)
    reshape_array = to_array.reshape((26,26))
    plt.figure(1)
    plt.imshow(reshape_array)
    plt.savefig(pic_path,format='png', bbox_inches='tight', transparent=True)
    plt.close()

def daw_28(data,pic_path):
    to_array = np.array(data)
    reshape_array = to_array.reshape((28,28))
    plt.figure(1)
    plt.imshow(reshape_array)
    plt.savefig(pic_path,format='png', bbox_inches='tight', transparent=True)
    plt.close()
def daw_1929(data,pic_path):
    to_array = np.array(data)
    reshape_array = to_array.reshape((19,29))
    plt.figure(1)
    plt.imshow(reshape_array)
    plt.savefig(pic_path,format='png', bbox_inches='tight', transparent=True)
    plt.close()
def daw_39(data,pic_path):

    to_array = np.array(data)
    reshape_array = to_array.reshape((39,39))
    plt.figure(1)
    plt.imshow(reshape_array)
    plt.savefig(pic_path,format='png', bbox_inches='tight', transparent=True)
    plt.close()
def daw_45(data,pic_path):
    to_array = np.array(data)
    reshape_array = to_array.reshape((30,45))
    plt.figure(1)
    plt.imshow(reshape_array)
    plt.savefig(pic_path,format='png', bbox_inches='tight', transparent=True)
    plt.close()
def daw_48(data,pic_path):

    to_array = np.array(data)
    reshape_array = to_array.reshape((48,48))
    plt.figure(1)
    plt.imshow(reshape_array)
    plt.savefig(pic_path,format='png', bbox_inches='tight', transparent=True)
    plt.close()
def daw_49(data,pic_path):

    to_array = np.array(data)
    reshape_array = to_array.reshape((49,49))
    plt.figure(1)
    plt.imshow(reshape_array)
    plt.savefig(pic_path,format='png', bbox_inches='tight', transparent=True)
    plt.close()
def daw_50(data,pic_path):
    to_array = np.array(data)
    reshape_array = to_array.reshape((50,50))
    plt.figure(1)
    plt.imshow(reshape_array)
    plt.savefig(pic_path,format='png', bbox_inches='tight', transparent=True)
    plt.close()
def daw_54(data,pic_path):
    to_array = np.array(data)
    reshape_array = to_array.reshape((54,54))
    plt.figure(1)
    plt.imshow(reshape_array)
    plt.savefig(pic_path,format='png', bbox_inches='tight', transparent=True)
    plt.close()
def daw_3959(data,pic_path):
    to_array = np.array(data)
    reshape_array = to_array.reshape((39,59))
    plt.figure(1)
    plt.imshow(reshape_array)
    plt.savefig(pic_path,format='png', bbox_inches='tight', transparent=True)
    plt.close()
def daw_69(data,pic_path):
    to_array = np.array(data)
    reshape_array = to_array.reshape((60,90))
    plt.figure(1)
    plt.imshow(reshape_array)
    plt.savefig(pic_path,format='png', bbox_inches='tight', transparent=True)
    plt.close()
def daw_84(data,pic_path):

    to_array = np.array(data)
    reshape_array = to_array.reshape((54,84))
    plt.figure(1)
    plt.imshow(reshape_array)
    plt.savefig(pic_path,format='png', bbox_inches='tight', transparent=True)
    plt.close()
def daw_90(data,pic_path):
    to_array = np.array(data)
    reshape_array = to_array.reshape((90,135))
    plt.figure(1)
    plt.imshow(reshape_array)
    plt.savefig(pic_path,format='png', bbox_inches='tight', transparent=True)
    plt.close()
def daw_104(data,pic_path):
    to_array = np.array(data)
    reshape_array = to_array.reshape((104,104))
    plt.figure(1)
    plt.imshow(reshape_array)
    plt.savefig(pic_path,format='png', bbox_inches='tight', transparent=True)
    plt.close()
def dir_tif(path,allfile):
    filelist =  os.listdir(path)
    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            dir_tif(filepath, allfile)
        else:
            tp = filepath.split('.')
            length_tp = len(tp)

            if tp[length_tp-1] != 'tif':
                pass
            else:
                allfile.append(filepath)
    return allfile

#from sklearn.decomposition import PCA
#暂时读取三个训练数据集数据(这个函数针对的数据大小是240*360的数据集)
def read_all_Train_Datas(model_new,path_root="/home/gzs/Documents/denys/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train/Train"):
    allvid = []
#    path_root = "/home/gzs/Documents/denys/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train/Train"
    for p in range(1,2):#17 8
        path = path_root+get_path_only(p)
        allfile = dir_tif(path,[])
        length = len(allfile)
        vid = []
        for i in range(1,length+1):
            tif_path = os.path.join(path,get_path_only(i)+".tif")
            img = cv2.imread(tif_path)
            img = np.asarray(cv2.resize(img,(360,240)),'f')
            img = img - np.mean(img)
            t_X = np.array(img)
            vid.append(t_X)
        vid = np.array(vid, dtype=np.float32)
        start_frame = 9
        for j in range(0,length-9):
            X=vid[start_frame-9:start_frame, :, :,:]
            X = np.expand_dims(X,axis=0)
            output = model_new.predict(np.array(X))[0,:,:,:]
            allvid.append(output)
            start_frame = start_frame+1
    allvid=np.array(allvid)
    return allvid
def read_frames_test(model_new,vid,start_frame):#测试数据集数据获取

    X = vid[start_frame-9:(start_frame), :, :, :]
#    X = X[:, 8:120, 30:142,:]
#    X = X[:, 8:232, 68:292,:]
    X = np.expand_dims(X,axis=0)
    output = model_new.predict(np.array(X))[0,:,:,:]

    return output
def get_path(index):

    if index>200:
        return
    else:
        #计算当前帧的前六帧的两两平均帧
        if index<10:
           path0 = '00'+str(index)+'.tif'

        elif(index>=10 and index< 100):
           path0 = '0'+str(index)+'.tif'

        elif(index>=100 and index<1000):

           path0 = str(index)+'.tif'
        else:
            return
    return path0

def get_path_only(index):
    if index>200:
        return
    else:
        if index<10:
           path0 = '00'+str(index)

        elif(index>=10 and index< 100):
           path0 = '0'+str(index)

        elif(index>=100 and index<1000):

           path0 = str(index)
        else:
            return
    return path0

if __name__ == '__main__':
    print("Tools Main Function is calling！")
    path_root = "/home/gzs/Documents/denys/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train/Train"
    for p in range(1,2):#17 8
        path = path_root+get_path_only(p)
        allfile = dir_tif(path,[])
        length = len(allfile)
        for i in range(1,length+1):
            tif_path = os.path.join(path,get_path_only(i)+".tif")
            img = cv2.imread(tif_path)
            img = np.asarray(cv2.resize(img,(360,240)),'f')
            print(img.shape)
            exit()
#        print(allfile)
#        exit()