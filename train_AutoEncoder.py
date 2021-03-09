# -*- coding: utf-8 -*-
"""
Created on %(date)s 2018 11 2
    训练自编码网络
@author: %(lys)s
"""
import cv2,numpy as np

np.set_printoptions(threshold='nan')

import os

#import math

#import calc_mean as cm

#import mahalanobis as mls

#import Tools as tl

#import matplotlib.pyplot as plt
#import generate_train as gt

import c3d_model_keras as spnet
from  deny_callback import history,save
import keras

import generate_train_auto_loop as gtal
from keras import optimizers

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#from keras import losses

def main():
#    pre_model = spnet.pre_weights_c3d_net(summary=False)#得到预训练权重
    model = spnet.AutoEncoder_Net()#构建自编码
    '''
    第一次训练了150epoch  le=1e-4  BATCH_SIZE = 1
    '''
#    model.load_weights("/media/gzs/denys/paper/models_1/weights-train-139-0.72206.hdf5")
#    model.load_weights("/media/gzs/denys/paper/models/weights-train-80-0.64267.hdf5")

#    nadam = keras.optimizers.Nadam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    sgd = optimizers.SGD(lr=1e-5, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd, loss='mse',metrics=['acc'])

    HISTORY = history()

    filepath="/media/gzs/denys/paper/models/models/weights-train-{epoch:02d}-{acc:.5f}.hdf5"

    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='acc',mode='auto', verbose=1, save_best_only=False,period=1)

    BATCH_SIZE  = 1
    base_path = "/media/gzs/denys/paper/auto_train/pedestrian2/"
    lines = gtal.list_all_files(base_path)

    per_epoch = len(lines)/BATCH_SIZE

    model.fit_generator(generator = gtal.generate_data_from_train(batch_size = BATCH_SIZE,rootdir = base_path),
                                samples_per_epoch=per_epoch,#per_epoch
                                epochs=150,
                                verbose = 1,
                                callbacks = [HISTORY,checkpoint])
#    print(model.summary())
    model.save_weights("./models/finally/auto_models.hdf5")
    save(HISTORY)

if __name__ == "__main__":
    main()