# -*- coding: utf-8 -*-

import numpy as np

'''
生成自编码训练数据集
'''

import c3d_model_keras as spnet

save_path = "/media/gzs/denys/auto_train/"
pedestrian1 = "/home/gzs/Documents/denys/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train/Train"

import cv2
import os
import sys
import math
import random
import matplotlib.pyplot as plt
import matplotlib.image as PImage
from PIL import Image


def get_file_path(index,base_path):
    if index > 99:
        return "None"
    if index < 10:
        path = base_path+"00"+str(index)
    else:
        path = base_path+"0"+str(index)
    return path
def get_img_name(index,length):
    if index > length+1:
        return "None"
    if index < 10:
        path = "00"+str(index)+".tif"
    elif index >9 and index < 100:
        path =  "0"+str(index)+".tif"
    elif index > 99:
        path = str(index)+".tif"
#    print(path)
    return path

def show_img(image):
    plt.figure(1)
    plt.imshow(image,cmap="gray")#
    plt.show()
def crop_size(image):
#    print(image.shape)

    img = image[8:232,1:]
#    print(img.shape)
    return img
def get_16_frame_image(index,length_img,base_path):
    datas = []
    for i in range(index-8,index+1):
        iname = get_img_name(i,length_img)
        tif_path = os.path.join(base_path,iname)
#        print(tif_path)
        img = np.asarray(cv2.imread(tif_path),'f')
#        img_tras = img.transpose((2,0,1))
#        print(img_tras.shape)
#        img_mean = np.mean(img_tras,keepdims=True)
#        print(img_mean)
#        exit()
        img = img - np.mean(img)
        img = crop_size(image=img)
        datas.append(img)
    return np.array(datas)
def main():
    # 224, 359
    pre_model = spnet.pre_weights_c3d_net(summary=True)#得到预训练权重
#    pre_model.predict(np.array(X))[0,:,:,:]
    file_list = [i for i in range(1,17)]

    random.shuffle(file_list)
    np_save_path_base = "/media/gzs/denys/paper/auto_train/pedestrian2/Train"
    for i in file_list:
        file_path = get_file_path(index=i,base_path=pedestrian1)
        files = os.listdir(file_path)
        length_img = len(files)
        index_all = length_img/9
        list_index = [lj for lj in range(1,index_all+1)]
#        random.shuffle(list_index)
        np_save_path = get_file_path(index=i,base_path=np_save_path_base)
        if os.path.exists(np_save_path):
            pass
        else:
            os.mkdir(np_save_path)
        for j in list_index:
            img_crop = get_16_frame_image(j*9,length_img,file_path)
            X = np.expand_dims(img_crop, axis=0)
            pre = pre_model.predict(np.array(X))[0,:,:,:]

            s_path = os.path.join(np_save_path,str(j)+".npy")

            if os.path.exists(s_path):
                os.remove(s_path)
                np.save(s_path,pre)
            else:
                np.save(s_path,pre)

if __name__ == "__main__":
    main()