# encoding:utf-8
import numpy as np
#import time
#import cv2
np.set_printoptions(threshold='nan')
import  os
import math
import matplotlib.pyplot as plt
#import threading

#import random
#补齐标准标签 0 1
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
# 得到第(index)几个测试集标准标注
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
def get_all_file(path):
    path = "./ROCOurMax/standrad/"
    rst = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        # print(file_path)
        rst.append(file_path)
    return np.array(rst)
def get_correct_count_standrad():
    path = "./ROCOurMax/standrad/"
    files = get_all_file(path)
    Count = 0
    for i in range(0,len(files)):
        tp_data = np.loadtxt(files[i])
        sum_tp = np.sum(tp_data==1)
        Count = Count + sum_tp
    return Count
        # print(np.sum(tp_data==1))
def get_error_count_standrad():
    path = "./ROCOurMax/standrad/"
    files = get_all_file(path)
    Count = 0
    for i in range(0,len(files)):
        tp_data = np.loadtxt(files[i])
        sum_tp = np.sum(tp_data==0)
        # print(sum_tp)
        Count = Count + sum_tp
    return Count
#计算数据集的标准下标
def calc_standrad(index,length,i):

#    standard_path = "./flag_new.txt"
    standard_path = "./flag.txt"
    #Our_flag

    Section = get_standard(index,standard_path)
    if -1 == Section:
        print("can't get the Section")
    else:
        print("Section : "+str(Section))
        rst_data = complement_the_subscript(Section,length)
    if os.path.exists("./ROCOurMax"):
        pass
    else:
        os.mkdir("./ROCOurMax")
    if os.path.exists("./ROCOurMax/standrad"):
        pass
    else:
        os.mkdir("./ROCOurMax/standrad")
    np.savetxt("./ROCOurMax/standrad/"+str(length+5)+"_"+str(i)+".txt",rst_data)
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
#the function must be modify during this week,becouse the algorthm have a big defect
def readAndProcessData(index,npy_index,threshold):
    # str_32_64 = "64b"
    root_path = "/media/gzs/denys/paper/maha_datas/MasPicMOur/msjpg"
    # root_path = "E:/tensorflow/detectcar_project/specialTrain/datas/"+str(str_32_64)+"/MasPic/msjpg"
    path = root_path+str(index)+"/"+str(npy_index)+"msj.npy"
    data = np.array(np.load(path),np.float32)
#    data = np.mean(data.reshape((2,30,45)),axis=0, keepdims=True).reshape(30,45)
#    print(np.max(data.reshape((2,30,45)),axis=0, keepdims=True).shape)
    data = np.max(data.reshape((2,30,45)),axis=0, keepdims=True).reshape(30,45)
#    print(data.shape)
#    exit()
    circle_count = 0
    list_point = []
    if index>9:
        path112 = "/home/gzs/Documents/denys/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test0"+str(index)+"/"+str(get_path_only(npy_index))+".tif"
    else:
        path112 = "/home/gzs/Documents/denys/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test00"+str(index)+"/"+str(get_path_only(npy_index))+".tif"
#    im = cv2.imread(path112,1)

    for i in range(0,30):
        for j in range(0,45):
            tp_data = data[i,j]
            if tp_data>=threshold:
                i_start = int(8*i)
                i_end = int(8*i+46)
                j_start = int(8*j)
                j_end = int(8*j+46)
                circle_count = circle_count + 1
                tmp = [(i_start+i_end)/2,(j_start+j_end)/2]
                list_point.append(tmp)
            else:
                pass
    if circle_count>=2:
         rst_point =  handle_anomaly_points(list_point)
         if len(rst_point)>0:
#            save_path = "./ROCOur/"+str(threshold)+"/"+str(index)+"/"+str(npy_index)+".png"
#            for f in range(0,len(rst_point)):
#                 j_start = int(rst_point[f][1]-25)
#                 j_end = int(rst_point[f][1]+25)
#                 i_start = int(rst_point[f][0]-25)
#                 i_end = int(rst_point[f][0]+25)
#                 cv2.rectangle(im,(j_start,i_start),(j_end,i_end),(0,0,255),1)# 猩红
#            cv2.imwrite(save_path,im)
            return 1
         else:
             return 0
    else:
#         im = cv2.imread(path112,1)
#         save_path = "./ROC/"+str(threshold)+"/"+str(index)+"/"+str(npy_index)+".png"
#         cv2.imwrite(save_path,im)
         return  0
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
def distence_points(points):
    length = len(points)
    rst_point = []
    start_point = []
    end_point = []
    for i in range(0,length-1):
        for j in range(i+1,length):
            start_x = points[i][0]
            start_y = points[i][1]
            end_x = points[j][0]
            end_y = points[j][1]
            span_x = start_x-end_x
            span_y = start_y-end_y
            distence_tp = math.sqrt(span_x*span_x+span_y*span_y)
#            print("distence_tp  :"+str(distence_tp))
            if distence_tp<=65:
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


def handle_anomaly_points(points):
     rst_point = distence_points(points)
     return  rst_point
#TP 预测是异常 标签也是异常
#FP 预测是异常  标签是非异常
#TN 预测是非异常 标签是非异常
#FN 预测是非异常 标签是异常
#0:非异常   1:异常
                 #预测  标签
# # global TP# = 0# 1    1
# TP = 0
# # global FP# = 0# 1    0
# FP = 0
# # global TN# = 0# 0    0
# TN = 0
# # global FN# = 0# 0    1
# FN = 0
# # global FPR
# FPR = -1000
# # global TPR
# TPR = -1000
def calc_tp_fn_fp_tn(threshold,index,length):

    Prediction_path = "./ROCOurMax/"+str(threshold)+"/anomaly_"+str(index)+".txt"

    standrad_path = "./ROCOurMax/standrad/"+str(length+5)+"_"+str(index)+".txt"
    # print("standrad_path : "+standrad_path)
    predic_data   = (np.loadtxt(Prediction_path)).astype(int)
    standrad_data = (np.loadtxt(standrad_path)).astype(int)
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    el = 0
    # print("predic_data  len : "+str(len(predic_data)))
    # print("standrad_data  len : "+str(len(standrad_data)))
    # print("length : "+str(length))
    for i in range(0,length):
        if predic_data[i] == 1 and standrad_data[i] == 1:#检测是一场实际也是异常
            TP = TP + 1#真正
        elif predic_data[i] == 1 and standrad_data[i] == 0:#没有异常说是有异常
            FP = FP + 1#误报
        elif predic_data[i] == 0 and standrad_data[i] == 0:#检测没有异常世实际也没有异常
            TN = TN + 1#真负
        elif predic_data[i] == 0 and standrad_data[i] == 1:#检测没有异常实际是有异常的
            FN = FN + 1#漏报
        el = el + 1
    if FP > 0:
        print(Prediction_path)
    return TP,FP,TN,FN,el
#index : the order of test data  set
#length : the length of current test data
def calc_anomaly(index,threshold,length):
    rst_data = []
    for i in range(9,length+9):
        tp = readAndProcessData(index,i,threshold)
        rst_data.append(int(tp))
    return rst_data
def Draw_Roc(tp,fp,tn,fn,el):#,count_correct,count_error):

    tpr = []
    fpr = []
    length = len(tp)
    for i in range(0,length,1):
        a = float(float(tp[i])/(float(tp[i])+float(fn[i])))
        b = float(float(fp[i])/(float(fp[i])+float(tn[i])))
        tpr.append(a)#+0.01
#        tp_fn = fn[i]+fp[i]
        print("tpr:"+str(a)+" tp="+str(tp[i])+" fpr: "+str(b)+" fp = "+str(fp[i]))
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
def fn_tn_tp_fp(threshold,index,length):

    tp = []
    fp = []
    tn = []
    fn = []
    EL = []
    i = 0
    Count = [n for n in range(0,len(threshold))]
    for Ct in Count:
        TP_1 = 0
        FP_1 = 0
        TN_1 = 0
        FN_1 = 0
        el_1 = 0
        for i in range(1,len(index)+1):
           # print("i : "+str(i))
           TP,FP,TN,FN,el = calc_tp_fn_fp_tn(threshold[Ct],index[i-1],length[i-1])
           if FP <=0:
               print("index[i-1] {0},threshold[Ct] {1},length[i-1] {2}".format(index[i-1],threshold[Ct],length[i-1]))
           TP_1 = TP_1 + TP
           FP_1 = FP_1 + FP
           TN_1 = TN_1 + TN
           FN_1 = FN_1 + FN
           el_1 = el_1 + el
        tp.append(TP_1)
        fp.append(FP_1)
        tn.append(TN_1)
        fn.append(FN_1)
        EL.append(el_1)

    return tp,fp,tn,fn,EL
def calc_anomaly_single(index,length,thres):
    if os.path.exists("./ROCOurMax/"+str(thres)):
        pass
    else:
        os.mkdir("./ROCOurMax/"+str(thres))

    if os.path.exists("./ROCOurMax/"+str(thres)+"/"+str(index)):
        pass
    else:
        os.mkdir("./ROCOurMax/"+str(thres)+"/"+str(index))
    rst_data = calc_anomaly(index,thres,length)
    np.savetxt("./ROCOurMax/"+str(thres)+"/anomaly_"+str(index)+".txt",rst_data)
    # print("thres : "+str(thres)+"  i :"+str(index))
#def calc_anomaly_thr(index,length):
#    threads_thr = []
#    for i in range(0, len(threshold)):
#        th = threading.Thread(target=calc_anomaly_single, args=(index,length,threshold[i]))
#        th.start()
#        threads_thr.append(th)
#    for th in threads_thr:
#        th.join()
#def threading_calc_anomly(index,length):
#    #定义线程池
#    threads = []
#    #先创建线程对象
#    for i in range(0, len(index)):
#        th = threading.Thread(target=calc_anomaly_thr, args=(index[i],length[i]))
#        th.start()
#        threads.append(th)
#    # 主线程中等待所有子线程退出
#    for th in threads:
#        th.join()
from sklearn.metrics import roc_curve, auc
if __name__ == '__main__':

    global threshold#

#    threshold = [i for i in np.arange(10,150,1)]
#    threshold = [i for i in np.arange(10,50,1)]
    threshold = [i for i in np.arange(5,100,1)]#5

    index = [1,2,3,4,5,6,7,8,9,10,11,12]

    length = [172,172,142,172,142,172,172,172,112,142,172,172]

    #准备标准标签
#    for i in range(1,13):#13
#         calc_standrad(index[i-1],length[i-1],i)
#    exit()
#     准备异常数据集
#    threading_calc_anomly(index,length)
#    for i in range(0,12):#12
#        for j in range(0,len(threshold)):
#             calc_anomaly_single(index[i],length[i],threshold[j])

    tp,fp,tn,fn,el = fn_tn_tp_fp(threshold,index,length)
    print("fp len {0}".format(len(tp)))
    exit()
    print("tp {0},fp {1},tn {2},fn {3}".format(tp,fp,tn,fn))
    print(" 预测为异常正确个数（真正tp）: "+str(tp))

    Draw_Roc(tp,fp,tn,fn,el)#,count_correct,count_error)