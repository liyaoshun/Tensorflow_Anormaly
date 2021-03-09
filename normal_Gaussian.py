#!/usr/bin/env python
# -*- coding: utf-8 -*-
#建立高斯正常行为的高斯模型

'''
2018 11 2 最后的论文实验归档 add
'''

import cv2,numpy as np

np.set_printoptions(threshold='nan')

import os

import math

import calc_mean as cm

import mahalanobis as mls

import Tools as tl

import matplotlib.pyplot as plt


import c3d_model_keras as spnet


import datetime
#生成马氏距离值和绘制马氏距离值得图
def main(T_Index,test_countt,data,model_new,mean,Matrix_cov):

#    ms = "MasPicMOurMax/msjpg"+str(T_Index)
    ms = "MasPicMOurPed/msjpg"+str(T_Index)
    ##########################测试集数据获取##########################
    if T_Index>9:
        path_dir_test = "/home/gzs/Documents/denys/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test0"+str(T_Index)+"/"
    else:
        path_dir_test = "/home/gzs/Documents/denys/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test00"+str(T_Index)+"/"
    vid = []
    for i in range(1,test_countt+1):
        sub_path = tl.get_path(i)
        path=path_dir_test+sub_path
        img = cv2.imread(path)
#        img = np.asarray(cv2.resize(img,(171,128)),'f')
        img = np.asarray(cv2.resize(img,(360,240)),'f')
        img = img - np.mean(img)
        t_X = np.array(img)
        vid.append(t_X)
    vid = np.array(vid, dtype=np.float32)#/255
    ##########################测试集数据获取#########################
    for index in range(9,test_countt+1):
        data_test = tl.read_frames_test(model_new,vid,index)
        Matrix_test = np.array(data_test)
        step1 = datetime.datetime.now()
        tmp = Matrix_test.reshape(-1,512)#392*512
#        print("tmp  : "+str(tmp.shape))
        maha_array= []
        maha_arrays= []
        maha_array = mls.mahalanobis_modify(tmp,Matrix_cov,mean)
        print("maha_array  shape : "+str(maha_array.shape))
        maha_array0 = np.mean(maha_array, axis=0, keepdims=True)#column
        maha_array1 = np.mean(maha_array, axis=1, keepdims=True)#row
        print("maha_array0  shape : "+str(maha_array0.shape))
        print("maha_array1  shape : "+str(maha_array1.T.shape))
        maha_arrays.append(maha_array0)
        maha_arrays.append(maha_array1.T)
        maha_mean = np.mean(np.array(maha_arrays),axis=0, keepdims=True)
        print("maha_mean shape: "+str(maha_mean.shape))
#        exit()
        base_path = '/media/gzs/denys/paper/maha_datas/'
        if os.path.exists(base_path+ms):
            pass
        else:
            os.mkdir(base_path+ms)

        sub_path = base_path+ms+'/'+str(index)+'msj.png'

        np.save(base_path+ms+'/'+str(index)+'msj.npy',maha_mean)

        maha_mean = np.max(np.array(maha_mean).reshape((2,30,45)),axis=0, keepdims=True)
        print("maha_array  : "+str(maha_mean.shape))

        tl.daw_3959(maha_mean,sub_path)

        step2 = datetime.datetime.now()
        print(' 阶段 : '+str(T_Index)+"  i = "+str(index)+'  耗费时间  : '+str(step2-step1))
if __name__ == '__main__':

    index = [1,2,3,4,5,6,7,8,9,10,11,12]#[i for i in range(1,35)]#
    length = [180,180,180,180,150,180,180,180,120,150,180,180]#np.array([200]*34)#
#    print(len(length))
#    exit()
#    inputs = (9, 240, 360, 3)
    pre_model = spnet.pre_weights_c3d_net(summary=True)

    data = tl.read_all_Train_Datas(pre_model,path_root="/home/gzs/Documents/denys/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train/Train")#所有的训练数据
#    exit()
    Matrix_cov = []
    mean = []
    Matrix = np.array(data)
    print("Matrix Matrix shape : "+str(Matrix.shape))
    reshape_32 = Matrix.reshape(-1,512)
    mean = cm.sampleMean(reshape_32)
    print("mean mean shape : "+str(mean.shape))

    Matrix_reshape_T = reshape_32.T
    Matrix_cov = np.cov(Matrix_reshape_T)
    Matrix_cov = np.array(Matrix_cov)
    for i in range(1, len(index)+1):
        main(i,length[i-1],data,pre_model,mean,Matrix_cov)
'''
    这些代码是2018 11 2 日被注释的。
#if __name__ == '__main__':
#    index = [1,2,3,4,5,6,7,8,9,10,11,12]#
#    length = [180,180,150,180,150,180,180,180,120,150,180,180]
#
#    model = spnet.fin_c3d_our_Migration_pre(summary=False)
#
#    model_new = spnet.fin_c3d_our_Migration_pre_fin(model_fin=model,summary=False)
#
#    data = tl.read_all_Train_Datas(model_new,path_root="/home/gzs/Documents/denys/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train/Train")#所有的训练数据
#
#    Matrix_cov = []
#
#    mean = []
#
#    Matrix = np.array(data)
#    print("Matrix Matrix shape : "+str(Matrix.shape))
#
#    reshape_32 = Matrix.reshape(-1,512)
#    mean = cm.sampleMean(reshape_32)
#    print("mean mean shape : "+str(mean.shape))
#
#    Matrix_reshape_T = reshape_32.T
#    Matrix_cov = np.cov(Matrix_reshape_T)
#    Matrix_cov = np.array(Matrix_cov)
##
#    for i in range(1, len(index)+1):
##    i = 2
#        main(i,length[i-1],data,model_new,mean,Matrix_cov)
'''