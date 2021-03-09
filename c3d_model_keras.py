# coding: utf-8

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Deconvolution3D,Conv3D,Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Input
import os
from keras.models import model_from_json
import numpy as np
from keras.utils.vis_utils import plot_model

import h5py
from keras import backend as K


def get_layour_our():
    layers = []
    path = '../models/sports1M_weights_tf.h5'
    f = h5py.File(path,'r')
    list_k = f.keys()
    for k in list_k:
        layers.append(k)
    return layers
#直接从h5文件中读取卷基层权重（C3D训练Sport1M）
def get_direct_h5_weights():
    path = '../models/sports1M_weights_tf.h5'
    f = h5py.File(path,'r')
    Kernels = []
    Biass = []
#    print(f['data'])
    list_k = f.keys()
    index = 1
    for k in list_k:
        if index > 8:
            return Kernels,Biass
        else:
            pass
        kk = f[k].keys()
        if len(kk)>0:
            print("keys  : "+str(kk[0]))
            data = f[k][kk[0]]
            gp_k = data.keys()
            Biass.append(data.get(gp_k[0]).value)
            Kernels.append(data.get(gp_k[1]).value)
#            print(str())
#            print(str(k)+"  bias  "+str(data.get(gp_k[0]).shape))
#            print(str(k)+"  kernel " +str(data.get(gp_k[1]).shape))
        else:
            pass
        index = index + 1
'''
2018 11 2 by lys // add
'''
def pre_weights_c3d_net(summary=False):
    Kernels,Biass = get_direct_h5_weights()
    print("Biass shape: "+str(np.array(Biass[0]).shape))
    print("kernels shape: "+str(np.array(Kernels[0]).shape))

    conv_weight = []
    for t in range(len(Kernels)):
        conv_weight.append([Kernels[t],Biass[t]])
    print("conv_weight len : "+str(len(conv_weight)))
    model = Sequential()
#    input_shape=(9, 224, 359, 3)# 9 224 358 3#自编码的时候使用的网络
    input_shape=(9, 240, 360, 3) # l, h, w, c

    model.add(Conv3D(64, (3, 3, 3), activation='relu',
                            padding='same', name='conv1',
                            input_shape=input_shape,weights=conv_weight[0]))#
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           padding='valid', name='pool1'))
    # 2nd layer group
    model.add(Conv3D(128, (3, 3, 3), activation='relu',
                            padding='same', name='conv2',weights=conv_weight[1]))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool2'))
    # 3rd layer group
    model.add(Conv3D(256, (3, 3, 3), activation='relu',
                            padding='same', name='conv3a',weights=conv_weight[2]))
    model.add(Conv3D(256, (3, 3, 3), activation='relu',
                            padding='same', name='conv3b',weights=conv_weight[3]))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool3'))
    # 4th layer group
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                            padding='same', name='conv4a',weights=conv_weight[4]))
#    model.add(Conv3D(512, (3, 3, 3), activation='relu',
#                            padding='same', name='conv4b',weights=conv_weight[5]))

    model.get_layer('conv1').trainable = False
    model.get_layer('conv2').trainable = False
    model.get_layer('conv3a').trainable = False
    model.get_layer('conv3b').trainable = False
    model.get_layer('conv4a').trainable = False
#    model.get_layer('conv4b').trainable = False

    if summary:
        print(model.summary())
    return model
'''
2018 11 7 by lys // add
UCSD 行人数据集2网络 使用None代替
'''
def pre_weights_c3d_net_ped1(input_shape=(9, 158, 238, 3),summary=False):
    Kernels,Biass = get_direct_h5_weights()
    print("Biass shape: "+str(np.array(Biass[0]).shape))
    print("kernels shape: "+str(np.array(Kernels[0]).shape))

    conv_weight = []
    for t in range(len(Kernels)):
        conv_weight.append([Kernels[t],Biass[t]])
    print("conv_weight len : "+str(len(conv_weight)))
    model = Sequential()
#    input_shape=(9, 224, 359, 3)# 9 224 358 3#自编码的时候使用的网络
#    input_shape=(9, None, None, 3) # l, h, w, c
#    input_shape=(9, 240, 360, 3) # l, h, w, c
#    input_shape=(9, None, None, 3) # l, h, w, c

    model.add(Conv3D(64, (3, 3, 3), activation='relu',
                            padding='same', name='conv1',
                            input_shape=input_shape,weights=conv_weight[0]))#
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           padding='valid', name='pool1'))
    # 2nd layer group
    model.add(Conv3D(128, (3, 3, 3), activation='relu',
                            padding='same', name='conv2',weights=conv_weight[1]))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool2'))
    # 3rd layer group
    model.add(Conv3D(256, (3, 3, 3), activation='relu',
                            padding='same', name='conv3a',weights=conv_weight[2]))
    model.add(Conv3D(256, (3, 3, 3), activation='relu',
                            padding='same', name='conv3b',weights=conv_weight[3]))
#    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
#                           padding='valid', name='pool3'))
    # 4th layer group
#    model.add(Conv3D(512, (3, 3, 3), activation='relu',
#                            padding='same', name='conv4a',weights=conv_weight[4]))
#    model.add(Conv3D(512, (3, 3, 3), activation='relu',
#                            padding='same', name='conv4b',weights=conv_weight[5]))

    model.get_layer('conv1').trainable = False
    model.get_layer('conv2').trainable = False
    model.get_layer('conv3a').trainable = False
    model.get_layer('conv3b').trainable = False
#    model.get_layer('conv4a').trainable = False
#    model.get_layer('conv4b').trainable = False

    if summary:
        print(model.summary())
    return model
'''
   根据front网络后接autoEncoder构建自编码网络
'''

from keras import initializers

def Swish(x):
    return x*K.sigmoid(1.0*x)

def AutoEncoder_Net():

    input_shape=(2, 28, 44, 512)
    model = Sequential()#front_model#

    model.add(Conv3D(512, (2, 3, 3), activation='relu',
                            padding='same', name='atconv1',
                            input_shape=input_shape,
                            kernel_initializer=initializers.random_normal(stddev=0.01)))#
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                       padding='same', name='atpool1'))
    model.add(Conv3D(512, (1, 3, 3), activation='relu',
                            padding='same', name='atconv2',
                            kernel_initializer=initializers.random_normal(stddev=0.01)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                       padding='same', name='atpool2'))


    model.add(Conv3D(512, (1, 1, 1), activation='relu',
                            padding='same', name='atconv3',
                            kernel_initializer=initializers.random_normal(stddev=0.01)))


    model.add(Deconvolution3D(512, (2, 2, 2),strides=(2, 2, 2),
                              padding='same', name='at_up_pool1'))
    model.add(Conv3D(512, (1, 3, 3), activation='relu',
                            padding='same', name='at_up_conv1',
                            kernel_initializer=initializers.random_normal(stddev=0.01)))
    model.add(Deconvolution3D(512, (1, 2, 2),strides=(1, 2, 2),
                              padding='same', name='at_up_pool2'))
#
    model.add(Conv3D(512, (3, 3, 3), activation='relu',#Swish,#'softplus',
                        padding='same', name='at_up_conv2',
                        kernel_initializer=initializers.random_normal(stddev=0.01)))
    return model

def C3D_Auto_Net():
    C3D_Model = pre_weights_c3d_net()
    AUto_Model = AutoEncoder_Net()

    print(C3D_Model.summary())
    print(AUto_Model.summary())
def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

if __name__ == '__main__':
    model = pre_weights_c3d_net_ped1(summary=True)
#    print(model.summary())