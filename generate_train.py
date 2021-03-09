# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(lys)s
"""
import cv2,numpy as np
np.set_printoptions(threshold=np.nan)
import os
#import math
import random
import time
from keras.utils import np_utils
import PIL.Image as Image
def get_true_path(path):
    str_arr = path.split('/')
    length = len(str_arr)
    rst = ''
    for i in range(1,length):
        rst = rst +"/"+ str_arr[i]
    length = len(rst)
    end_char = rst[length-1]
    if end_char == '/':
        rst = rst[0:length-1]
    else:
        pass
    return rst
def read_jpg_from_disc(filename,num_frames_per_clip=9):
    ret_arr = []
    s_index = 0
    filenames = os.listdir(filename)
#     for file_name in filenames:
#     if(len(filenames)<num_frames_per_clip):
#         return [], s_index
    s_index = random.randint(1, len(filenames) - num_frames_per_clip)#从数据中随机开始位置取9帧数据
    if s_index>1000:
        s_index = int(s_index/2)
    for i in range(s_index, s_index + num_frames_per_clip):
        image_name = str(filename) + '/' + str(get_path_only(i))+".jpg"
        img = Image.open(image_name)
        img_data = np.array(img)-np.mean(np.array(img))
        ret_arr.append(img_data)
#    print(str(np.array(ret_arr).shape))
    return ret_arr
def read_jpg_from_disc_our(filename,num_frames_per_clip=9):
    ret_arr = []
    for i in range(1,10):
        image_name = str(filename) + '/' + str(i)+".jpg"
        img = Image.open(image_name)
        img_data = np.array(img)-np.mean(np.array(img))
        ret_arr.append(img_data)
    return ret_arr
def get_path_only(index):
    if index>10000:
        return
    else:
        #计算当前帧的前六帧的两两平均帧
        if index<10:
           path0 = '0000'+str(index)
        elif(index>=10 and index< 100):
           path0 = '000'+str(index)

        elif(index>=100 and index<1000):
           path0 = '00'+str(index)
        elif(index>=1000 and index<10000):
            path0 = '0'+str(index)
        else:
            return
    # print(path0)
    return path0
def crop(tmp_data):#9 240 360 3
     crop_size = 112
     tmp_data = np.array(tmp_data)
     img_datas = []
     for j in xrange(9):
         img = tmp_data[j,:,:,:]
         img = np.array(cv2.resize(np.array(img),(168, 112))).astype(np.float32)
#         crop_x = int((112 - crop_size)/2)
         crop_y = int((168 - crop_size)/2)
         img = img[:, crop_y:crop_y+crop_size,:]
         img_datas.append(img)
#     print("img_datas shape : "+str(np.array(img_datas).shape))
     return np.array(img_datas)
def generate_data_from_train(batch_size,category=101):
    X = []
    Y = []
    cnt=0
    lines=read_allfiles_labels_andindex(path='/media/gzs/datas/ucfTrainTestlist/trainlist01.txt')
    random.seed(time.time())
    video_indices = list(range(len(lines)))
    random.shuffle(video_indices)
    root = '/media/gzs/datas/UCF-101/'
    while 1:
        random.seed(time.time())
        random.shuffle(video_indices)
        for index in video_indices:
            tmp = lines[index].strip().split(" ")
            tp_label = int(tmp[1].strip()) - 1
            tmp_path_root = root+tmp[0].strip().split(".")[0]
            if int(tp_label) >= 0 and int(tp_label)<category:
                pass
            else:
                print('generate_data_from_train warning at generate_data 96,labels index : '+str(int(tp_label)))
                continue
            ret_arr = read_jpg_from_disc(filename=tmp_path_root)
            ret_arr_crop = crop(ret_arr)
            X.append(ret_arr_crop)
            Y.append(int(tp_label))
            cnt += 1
            if cnt==batch_size:
                cnt = 0
                y_train = np_utils.to_categorical(Y,category)
#                print("generate_data running yield")
                yield  (np.array(X),np.array(y_train))
                X = []
                Y = []
def generate_data_from_train_our(batch_size,category=5):
    X = []
    Y = []
    cnt=0
    lines=read_allfiles_labels_andindex(path='/home/gzs/Documents/denys/C3D_Anormaly_keras/train_fin/out_train.txt')
    random.seed(time.time())
    video_indices = list(range(len(lines)))
    random.shuffle(video_indices)
#    root = '/media/gzs/datas/UCF-101/'
    while 1:
        random.seed(time.time())
        random.shuffle(video_indices)
        for index in video_indices:
            tmp = lines[index].strip().split("\t")
            tp_label = int(tmp[1].strip())# - 1
#            tmp_path_root = root+tmp[0].strip().split(".")[0]
            tmp_path_root = tmp[0].strip().split(".")[0]
            if int(tp_label) >= 0 and int(tp_label)<category:
                pass
            else:
                print('generate_data_from_train warning at generate_data 96,labels index : '+str(int(tp_label)))
                continue
            ret_arr = read_jpg_from_disc_our(filename=tmp_path_root)
#            ret_arr_crop = crop(ret_arr)
            X.append(ret_arr)
            Y.append(int(tp_label))
            cnt += 1
            if cnt==batch_size:
                cnt = 0
                y_train = np_utils.to_categorical(Y,category)
#                print("generate_data running yield")
                yield  (np.array(X),np.array(y_train))
                X = []
                Y = []
def get_data_to_test(batch_size,category=101):
    X = []
    Y = []
    cnt=0
    lines=read_allfiles_labels_andindex(path='/media/gzs/datas/ucfTrainTestlist/testlist01.txt')
    random.seed(time.time())
    video_indices = list(range(len(lines)))
    random.shuffle(video_indices)
    root = '/media/gzs/datas/UCF-101/'
    while 1:
        random.seed(time.time())
        random.shuffle(video_indices)
        for index in video_indices:
            tmp = lines[index].strip().split(" ")
            tp_label = int(tmp[1].strip()) - 1
            tmp_path_root = root+tmp[0].strip().split(".")[0]
            if int(tp_label) >= 0 and int(tp_label)<category:
                pass
            else:
                print('generate_data_from_train warning at generate_data 96,labels index : '+str(int(tp_label)))
                continue
            ret_arr = read_jpg_from_disc(filename=tmp_path_root)
            ret_arr_crop = crop(ret_arr)
            X.append(ret_arr_crop)
            Y.append(int(tp_label))
            cnt += 1
            if cnt==batch_size:
                cnt = 0
                y_train = np_utils.to_categorical(Y,category)
#                print("generate_data running yield")
                return  (np.array(X),np.array(y_train))
#                X = []
#                Y = []
def read_allfiles_labels_andindex(path):
    fr = open(path)
    lines = fr.readlines()
    return lines
#if __name__ == '__main__':
#    lines=read_allfiles_labels_andindex(path='/media/gzs/denys/ProcessingData/Datas/merge_data/valid_all.txt')
#    video_indices = list(range(len(lines)))
#    for index in video_indices:
#        tmp = lines[index].strip().split("\t")
#        tp_label = tmp[1].strip()
#        tmp_path_root = tmp[0].strip()
##        print("tp_label  " +str(tp_label)+"   tmp_path_root  "+str(tmp_path_root))
##        exit()
#        read_jpg_from_disc('/media/gzs/denys/ProcessingData',tmp_path_root)#read data and clip the data
#    filename = '/media/gzs/datas/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01'
#    ret_arr, s_index = read_jpg_from_disc(filename=filename,tmp_label=0)
#    crop(ret_arr)
#    generate_data_from_train(batch_size=80)
#    lines=read_allfiles_labels_andindex(path='/media/gzs/datas/ucfTrainTestlist/trainlist01.txt')
#    print(len(lines))
#    tmp = lines[0].strip().split(" ")
##    print("tmp  : "+str(tmp))
#    tp_label = tmp[1].strip()
#    tmp_path_root = tmp[0].strip()