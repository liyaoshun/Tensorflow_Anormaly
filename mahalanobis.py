# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 15:29 2017
@function 马氏距离计算
         协方差矩阵计算的是不同维度之间的协方差，而不是不同样本之间的。
         理解协方差矩阵的关键就在于牢记它计算的是不同维度之间的协方差，
         而不是不同样本之间，拿到一个样本矩阵，我们最先要明确的就是一行是一个样本还是一个维度，
         心中明确这个整个计算过程就会顺流而下，这么一来就不会迷茫了~
@author: 李谣顺
"""
import numpy as np
from numpy import *
# array 测试数据的均值矩阵
# Mean  样本总体的均值
# var1  总体训练样本的协方差矩阵
def mahalanobis_modify(array,var1,Mean):
#    print("array shape :"+str(array.shape))#529*256
#    print("var1 shape :"+str(var1.shape))#256*256
#    print("Mean shape :"+str(Mean.shape))#1*256
#    array_line = []
    
#    array_line = [array]
#    sio.savemat('saveddata.mat', {'data': var1})#保存matlab数据格式
    var = np.matrix(var1)
#    tl.daw(var,(256,256))
    var1_inv =  var.I#np.linalg.pinv(var)#

    Array_Mean = np.array(Mean)

    array = np.array(array)

    lef_matrix = array-Array_Mean

    rig_matrix = array.T-Array_Mean.T
    # print("rig_matrix : "+str(rig_matrix.shape))

    lef_matrix_varinv = lef_matrix*var1_inv
#    print("lef_matrix : "+str(lef_matrix.shape))
#    print("lef_matrix_varinv : "+str(lef_matrix_varinv.shape))
#    print("rig_matrix : "+str(rig_matrix.shape))
    lef_matrix_varinv_rig = lef_matrix_varinv*(rig_matrix)

    G_Value = lef_matrix_varinv_rig

    return np.array(G_Value)
# array 测试数据的均值矩阵
# Mean  样本总体的均值
# var1  总体训练样本的协方差矩阵
def mahalanobis(array,var1,Mean):

    array_line = []
    
    array_line = [array]

    var = np.matrix(var1)

    var1_inv = var.I#np.linalg.pinv(var)#

    Array_Mean = np.array(Mean)

    array_line = np.array(array_line)

    lef_matrix = array_line-Array_Mean

    rig_matrix = array_line.T-Array_Mean.T
    # print("rig_matrix : "+str(rig_matrix.shape))

    lef_matrix_varinv = lef_matrix*var1_inv
    # print("lef_matrix_varinv : "+str(lef_matrix_varinv.shape))
    lef_matrix_varinv_rig = lef_matrix_varinv*(rig_matrix.reshape(-1,1))

    G_Value = lef_matrix_varinv_rig

    return G_Value