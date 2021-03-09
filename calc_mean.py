# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 10:37 2017
@function 总体样本的均值
@author: 李谣顺
"""
import cv2,numpy as np
import os

import math



def sampleMean(data):
    Sum_Mean = np.mean(data, axis=0, keepdims=True)#axis=0 列 axis=1 行
    return Sum_Mean