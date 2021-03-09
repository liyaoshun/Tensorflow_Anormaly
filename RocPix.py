#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 16:07:31 2018
在行人数据集2上的像素集的ROC计算
@author: lys
"""

import numpy as np
#import time
import cv2
np.set_printoptions(threshold='nan')
import  os
import math
import matplotlib.pyplot as plt
#import threading

#from sklearn.metrics import roc_curve, auc 

ROC_save_path = "RocMaxPix"

def our_contains(list,t):
    if 0==len(list):
        return False
    for i in range(0,len(list)):
        if t[0] == list[i][0] and  t[1] == list[i][1]:
            return True
        elif t[0] == list[i][1] and  t[1] == list[i][0]:
            return True
        else:
            return False
'''
points : 表示预测出来的异常区域中心点
return rst_point H,W
'''
def distence_points(points):
    length = len(points)
    rst_point = []

    start_point = []
    end_point = []

    for i in range(0,length-1):
        for j in range(i+1,length):
            start_x = points[i][0]#H
            start_y = points[i][1]#W
            end_x = points[j][0]
            end_y = points[j][1]
            
            span_x = abs(start_x-end_x)#H
            span_y = abs(start_y-end_y)#W
            distence_tp = math.sqrt(span_x*span_x+span_y*span_y)
            if distence_tp<=15:

                start_point = [start_x,start_y]
                end_point = [end_x,end_y]
                if our_contains(rst_point,start_point):#rst_point.contains(start_point):
                    pass
                else:
                    rst_point.append(start_point)
                if our_contains(rst_point,end_point):#rst_point.contains(end_point):
                    pass
                else:
                    rst_point.append(end_point)
            else:
                pass
    return  rst_point
'''
points H,W
'''
def handle_anomaly_points(points):
     rst_point = distence_points(points)
     return  rst_point
 
def get_xml_name(np_index):
    
    assert np_index <=200
    if np_index < 10:
        return "00"+str(np_index)+".xml"
    elif np_index >=10 and np_index < 100:
        return "0"+str(np_index)+".xml"
    else:
        return str(np_index)+".xml"
#from xml.etree.ElementTree import ElementTree, Element
#def get_xml_points(path):
#    rst = []
#    tree = ElementTree()
#    tree.parse(path)
#    
#    return rst
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
import matplotlib.patches as patches
def draw_rect(index,npy_index,x,y,w,h,lab_xmin,lab_ymin,lab_xmax,lab_ymax,iou):
    path = "/home/gzs/Documents/denys/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test"+get_path_only(index)+"/"+get_path_only(npy_index)+".tif"
    print("path : "+path)
    img = cv2.imread(path,1)
    plt.figure(1)
    plt.imshow(img)
    currentAxis=plt.gca()
    rect=patches.Rectangle((x, y),w,h,linewidth=1,edgecolor='yellow',facecolor='none')
    rectlabel=patches.Rectangle((lab_xmin, lab_ymin),lab_xmax-lab_xmin,lab_ymax-lab_ymin,linewidth=1,edgecolor='blueViolet',facecolor='none')
    currentAxis.add_patch(rect)
    currentAxis.add_patch(rectlabel)
    plt.title("iou area {0},test00{1},index {2}.tif".format(iou,index,npy_index))
    plt.show()
def draw_rect_no_content(index,npy_index,x,y,w,h,lab_xmin,lab_ymin,lab_xmax,lab_ymax):
    path = "/home/gzs/Documents/denys/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test"+get_path_only(index)+"/"+get_path_only(npy_index)+".tif"
    print("path : "+path)
    img = cv2.imread(path,1)
    plt.figure(1)
    plt.imshow(img)
    currentAxis=plt.gca()
    rect=patches.Rectangle((x, y),w,h,linewidth=1,edgecolor='yellow',facecolor='none')
    rectlabel=patches.Rectangle((lab_xmin, lab_ymin),lab_xmax-lab_xmin,lab_ymax-lab_ymin,linewidth=1,edgecolor='blueViolet',facecolor='none')
    currentAxis.add_patch(rect)
    currentAxis.add_patch(rectlabel)
    plt.title("test00{0},index {1}.tif".format(index,npy_index))
    plt.show()
#    exit()
'''
判断两个矩形是否相交
'''
def mat_inter(box1,box2):
    # box=(xA,yA,xB,yB)
    x01, y01, x02, y02 = box1
    x11, y11, x12, y12 = box2

    lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
    ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
    sax = abs(x01 - x02)
    sbx = abs(x11 - x12)
    say = abs(y01 - y02)
    sby = abs(y11 - y12)
    if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:
        return True
    else:
        return False
'''
[x1, y1, x2, y2] 
其中 (x1, y1) 为左上角的坐标，(x2, y2) 是右下角的坐标
'''
def is_cover(rec1,rec2):
    if np.maximum(rec1[0],rec2[0])<np.minimum(rec1[2],rec2[2]) and np.maximum(rec1[1],rec2[1])<np.minimum(rec1[3],rec2[3]):
        return True
    else:
        False
'''
计算两个矩形框的重合度
'''
def solve_coincide(box1,box2):
    # box=(xA,yA,xB,yB)
    # 
    if mat_inter(box1,box2)==True:
        x01, y01, x02, y02 = box1
        x11, y11, x12, y12 = box2
        col=min(x02,x12)-max(x01,x11)
        row=min(y02,y12)-max(y01,y11)
        intersection=col*row
        area1=(x02-x01)*(y02-y01)
#        area2=(x12-x11)*(y12-y11)
#        coincide=intersection/(area1+area2-intersection)
        coincide=intersection/area1
        return coincide
    else:
        return False
def judge_array_exits(alllist,tmp):
    length = len(alllist)
    for i in range(0,length):
        if (tmp == np.array(alllist[i])).all():
            return True,i
        else:
            continue
    return False,i
'''
index : 表明第几个测试集
npy_index : 表明当前测试集的第几个图像
list_box  : 预测出来的异常框中心点（H,W）
w : 表示接受域的 1/2 大小
'''
def Box_IOU(index,npy_index,list_box,w = 23):
    label_base = "/home/gzs/Documents/denys/C3D_Anormaly_keras/paper/label/UCSDped2/test"+str(index)+"/"
    label_path = os.path.join(label_base,get_xml_name(npy_index))
#    print(label_path)
    is_Iou5 = False
#    Iou_ratio = 0
    if os.path.exists(label_path):
        pass
    else:
        return is_Iou5
    label_list = GetAnnotBoxLoc(label_path)['bicycle']#当前测试集第npy_index张图像的BBOX数据
#    img_path = "/home/gzs/Documents/denys/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test001/075.tif"
    lab_length = len(label_list)
    lst_length = len(list_box)
#    false_position = []#离散点的集合
    for j in range(0,lst_length):
        h_center = list_box[j][0]
        w_center = list_box[j][1]
        xmin = w_center - w
        ymin = h_center - w
        xmax = w_center + w
        ymax = h_center + w
        rect2 = [xmin,ymin,xmax,ymax]
        '''
        rect2_center  [x,y,w,h]  预测框数据
        '''
#        rect2_center = [(xmin+xmax)/2,(ymin+ymax)/2,(xmax-xmin),(ymax-ymin)]#x,y,w,h
        break_flag = False
        for i in range(0,lab_length):
            lab_xmin = label_list[i][0]#W
            lab_ymin = label_list[i][1]#H
            lab_xmax = label_list[i][2]#W
            lab_ymax = label_list[i][3]#H
            
            rect1 = [lab_xmin, lab_ymin, lab_xmax, lab_ymax] 
            cover = is_cover(rect1,rect2)
            
            if cover:#当有重叠的时候:
                
                Iou_ratio = solve_coincide(rect1,rect2)
                if Iou_ratio > 0.4:
                    break_flag = True
                    is_Iou5 = True
                    break
                else:
                    is_Iou5 = False
                
            else:
                continue
        if break_flag:
            break

    return is_Iou5
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
'''
index : 表示第几个数据集(测试集)
npy_index : 表示当前测试集中的第张图像
threshold : 阈值
'''
def readAndProcessData_modify(index,npy_index,threshold):
    root_path = "/media/gzs/denys/paper/maha_datas/MasPicMOur/msjpg"
    path = root_path+str(index)+"/"+str(npy_index)+"msj.npy"
    TP = 0# 预测是异常 实际是异常
    FP = 0# 预测是异常 实际是正常
    TN = 0# 预测是正常 实际是正常
    FN = 0# 预测是正常 实际是异常
    
    
#    standard_path = "./flag_new.txt"
#    Section = get_standard(index,standard_path)
    
#    left_index  = int(Section[0])
#    right_index = int(Section[1])
    
    label_base = "/home/gzs/Documents/denys/C3D_Anormaly_keras/paper/label/UCSDped2/test"+str(index)+"/"
    label_path = os.path.join(label_base,get_xml_name(npy_index))
    lb_nask = 0#0表示当前样本是正常的  1表示是异常的
    '''
    如果存在文件的话就说名是含有异常的图像样本
    没有文件的话就说明是正常的图像样本
    '''
#    if npy_index >=left_index and left_index <=right_index:
#        lb_nask = 1
#    else:
#        lb_nask = 0
    if os.path.exists(label_path):
        lb_nask = 1#异常样本
    else:
        lb_nask = 0#正常样本
    
    data = np.array(np.load(path),np.float32)
    data = np.max(data.reshape((2,30,45)),axis=0, keepdims=True).reshape(30,45)
    circle_count = 0
    list_point = []
    for i in range(0,30):
        for j in range(0,45):
            tp_data = data[i,j]
            if tp_data>=threshold:
                '''
                0.125 ： 是因为3次maxpool
                230 ： 是因为接受域计算出来是46
                '''
                i_start = i/0.125-23.0
                i_end =  i/0.125+23.0
                j_start = j/0.125-23.0
                j_end = j/0.125+23.0
                circle_count = circle_count + 1
                tmp = [(i_start+i_end)/2,(j_start+j_end)/2]#(H,W) 计算接受域的中心点
                list_point.append(tmp)
            else:
                pass
    if circle_count>=2:
         rst_point =  handle_anomaly_points(list_point)#H,W  #list_point#
         if len(rst_point)>0:
            '''
            像素级别的异常需要判断box和mask的IOU
            '''
            
            if lb_nask == 1:#当标签是异常的时候，需要去计算像素级别的mask
                is_positive = Box_IOU(index,npy_index,rst_point,w = 23)
                if is_positive:
                    #实际是正常 预测是异常
                    TP = 1
                else:
                    #实际是异常 预测是正常
                    FN = 1
            else:
                #标记的是正常 0  预测的是异常 1
                FP = 1
         else:
             pass
    else:# 预测的是正常 0
         if lb_nask == 1:
             #标记为异常。预测为正常
             FN = 1
         elif lb_nask == 0:
             #标记为正常 预测为正常
             TN = 1
             
    return TP,FP,TN,FN

def calc_anomaly(index,threshold,length):
#    rst_data = []
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(9,length+9):
        iTP,iFP,iTN,iFN = readAndProcessData_modify(index,i,threshold)
        TP = TP + iTP
        FP = FP + iFP
        TN = TN + iTN
        FN = FN + iFN
#        rst_data.append(int(tp))
    return TP,FP,TN,FN

'''
index 代表数据集的下标 如 Test001 中 index就为1
func:得到index下标测试集中异常帧的序列
return:如果返回为-1的时候就表示没有这个index测试集。正常返回的是开始帧和结束帧
'''
def get_standard(index,standard_path):
    f = open(standard_path)
    lines = f.readlines()
    record = 1
    for line in lines:
        if record == index:
            curLine = line.strip().split(':')
            return curLine
        else:
            record = record + 1
    return -1
'''
使用异常的左右游标和数据集长度来构成0、1的label。
eg:data=>[20,120] length=>172.返回的数据就是一个长为172的数组[0,19][121,171]为0,其他的为1
'''
def complement_the_subscript(data,length):
    left_index  = int(data[0])
    right_index = int(data[1])
    rst_data = []
    for i in range(9,length+9):
        if i<left_index or i>right_index:
            rst_data.append(0)
        else:
            rst_data.append(1)
    rst_data = np.array(rst_data)
    return rst_data
def calc_standrad(index,length,i):

    standard_path = "./flag_new.txt"
#    standard_path = "./Our_flag.txt"
    #Our_flag

    Section = get_standard(index,standard_path)
    if -1 == Section:
        print("can't get the Section")
        assert -1 == Section
    else:
        print("Section : "+str(Section))
        rst_data = complement_the_subscript(Section,length)
    if os.path.exists("./"+ROC_save_path):
        pass
    else:
        os.mkdir("./"+ROC_save_path)
    if os.path.exists("./"+ROC_save_path+"/standrad"):
        pass
    else:
        os.mkdir("./"+ROC_save_path+"/standrad")
    np.savetxt("./"+ROC_save_path+"/standrad/"+str(length+5)+"_"+str(i)+".txt",rst_data)
'''
index : 表示第几个数据集
length : 表示这个数据集的图像张数-8（因为从第九行开始计算的异常）
thres : 阈值
'''
def calc_anomaly_single(index,length,thres):
    if os.path.exists("./"+ROC_save_path+"/"+str(thres)):
        pass
    else:
        os.mkdir("./"+ROC_save_path+"/"+str(thres))

    if os.path.exists("./"+ROC_save_path+"/"+str(thres)+"/"+str(index)):
        pass
    else:
        os.mkdir("./"+ROC_save_path+"/"+str(thres)+"/"+str(index))
    TP,FP,TN,FN = calc_anomaly(index,thres,length)
    return TP,FP,TN,FN
#    np.savetxt("./"+ROC_save_path+"/"+str(thres)+"/anomaly_"+str(index)+".txt",rst_data)
def Draw_Roc(tp,fp,tn,fn):#,count_correct,count_error):

    tpr = []
    fpr = []
    length = len(tp)
    for i in range(0,length,1):
        a = float(float(tp[i])/(float(tp[i])+float(fn[i])))
        b = float(float(fp[i])/(float(fp[i])+float(tn[i])))
        tpr.append(a)#+0.01
#        tp_fn = fn[i]+fp[i]
#        print("tpr:"+str(a)+" tp="+str(tp[i])+" fpr: "+str(b)+" fp = "+str(fp[i])+" tp_fn:"+str(tp_fn))
        print("tp {0},fn {1},fp {2},tn {3}".format(tp[i],fn[i],fp[i],tn[i]))
        fpr.append(b)
    fpr = np.array(fpr)
    tpr = np.array(tpr)

    lines = plt.plot(fpr,tpr)
    plt.plot([0,0.5,1],[1,0.5,0],'go-', label='line 1', linewidth=1)
    plt.setp(lines, color='r', linewidth=1.0)
    plt.xlabel("fpr")
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.ylabel("tpr")
    plt.title("ROC")
    plt.show()
    
if __name__ == '__main__':

    global threshold#

#    threshold = [i for i in np.arange(5,100,1)]

    threshold = [i for i in np.arange(5,100,1)]

    index = [1,2,3,4,5,6,7,8,9,10,11,12]

    length = [172,172,142,172,142,172,172,172,112,142,172,172]
    #准备标准标签
    for i in range(1,13):#13
         calc_standrad(index[i-1],length[i-1],i)
#    exit()
#     准备异常数据集
#    threading_calc_anomly(index,length)
    tp = []
    fp = []
    tn = []
    fn = []
    for j in range(0,len(threshold)):
        tpj = 0
        fpj = 0
        tnj = 0
        fnj = 0
        for i in range(0,12):#12
             TP,FP,TN,FN = calc_anomaly_single(index[i],length[i],threshold[j])
             tpj += TP
             fpj += FP
             tnj += TN
             fnj += FN
        tp.append(tpj)
        fp.append(fpj)
        tn.append(tnj)
        fn.append(fnj)
    print("tp len {0} ".format(len(tp)))
#    exit()
    print(" 预测为异常正确个数（真正tp）: "+str(tp))

    Draw_Roc(tp,fp,tn,fn)#,count_correct,count_error)
    