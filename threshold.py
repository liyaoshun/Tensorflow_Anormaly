# encoding:utf-8
import numpy as np
import time
import cv2
np.set_printoptions(threshold='nan')
import  os
import PIL.Image as Image, PIL.ImageDraw as ImageDraw
import matplotlib.pyplot as plt

# import sys
# reload(sys)
# sys.setdefaultencoding('utf8')

#计算每一个视频片段的阈值
def main(index,start_index,end_index):
    root_path = "/media/gzs/denys/paper/maha_datas/MasPicMOur/msjpg"+str(index)+"/"
#    All_array = []
    max_x_array = []
    for i in range(start_index,end_index):
        path = root_path+str(100)+"msj.npy"
        
        tmp_data = np.load(path).reshape(2,30,45)

#        print("data  shape  : "+str(data.shape))
#        tmp_data = np.mean(data, axis=0, keepdims=True)
        #data, axis=0, keepdims=True
#        print("tmp data  shape  : "+str(tmp_data.shape))
        max_x = np.max(np.array(tmp_data))
        # print(str(max_x))
        re = np.where(tmp_data==max_x)
        # print(str(re[0][0])+"  "+str(re[1][0]))
        # if max_x>10000:
        #     continue
        max_x_array.append(max_x)
#        print("shape : "+str(tmp_data.shape))
        # exit()
#        tmp_data = np.reshape(tmp_data,1350)#4264 6889
#        All_array.append(tmp_data)#np.max(tmp_data)
    max_x_array = np.array(max_x_array)
    max_x_array.sort()
    # max_x_array = np.argsort(max_x_array)## 求All_array从小到大排序的坐标
    # print("    index  : "+str(index)+"      ssss    "+str(max_x_array))
    # All_array = np.argsort(All_array)## 求All_array从小到大排序的坐标
    # All_array = np.array(All_array)
    # print(str(All_array.shape))
#    cv2.imwrite("/media/gzs/denys/paper/maha_datas/MasPicMOur/threshold",np.reshape(max_x_array,()))
    np.save("/media/gzs/denys/paper/maha_datas/MasPicMOur/threshold/"+str(index)+"threshold.npy",max_x_array)
    draw_hist(index,max_x_array)

def draw_hist(index,datas):
    plt.hist(datas,100)#,100
    plt.xlabel('Current Count')
    plt.ylabel("mahalanobis distence value")
    plt.title('mahalanobis hist:  '+str(index))
    # plt.show()
    plt.savefig("/media/gzs/denys/paper/maha_datas/MasPicMOur/threshold/"+str(index)+"threshold.png")
    plt.close()
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
    # print(path0)
    return path0
#def cut_pic(T_Index,test_countt):
#    if T_Index>9:
#        path_dir_test = "E:/tensorflow/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test0"+str(T_Index)+"/"
#    else:
#        path_dir_test = "E:/tensorflow/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test00"+str(T_Index)+"/"
#
#    for i in range(1,test_countt+1):
#        sub_path = get_path(i)
#        path=path_dir_test+sub_path
#        print("path : "+str(path))
#        img = cv2.imread(path,1)
#        img = cv2.resize(img,(180,180))
#
#        img = img[8:120,30:142,:]
#        # print(img.shape)
#        # cv2.imshow("img",img)
#        # cv2.waitKey(0)
#        # exit()
#        if os.path.exists("../datas/cut/"+str(T_Index)):#exists
#            pass
#        else:
#            os.mkdir("../datas/cut/"+str(T_Index))
#        cv2.imwrite("../datas/cut/"+str(T_Index)+"/"+str(i)+".jpg",img)
if __name__ == '__main__':
    # index = 12
    # start_index = 9
    # end_index = 180
    # main(index,start_index,end_index)
    index = [1,2,3,4,5,6,7,8,9,10,11,12]#1,2,3,4,5,
    length = [180,180,150,180,150,180,180,180,120,150,180,180]#
    for i in index:
        main(i,9,length[i-1])
        # cut_pic(i,length[i-1])# 1 180# 2 180# 3 150#



