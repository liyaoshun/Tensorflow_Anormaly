# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(lys)s
"""

'''
在训练自编码阶段的输入和标签数据的生成脚本
'''
import random
import numpy as np
import time
import os
def list_all_files(rootdir):
#    import os
    _files = []
    list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
    for i in range(0,len(list)):
           path = os.path.join(rootdir,list[i])
           if os.path.isdir(path):
              _files.extend(list_all_files(path))
           if os.path.isfile(path):
              _files.append(path)
    return _files
#def read_all_files(base_path):
#    list_files = os.listdir(base_path)
#    print(list_files)
#    return

def generate_data_from_train(batch_size,rootdir):
    X = []
    Y = []
    cnt=0
    lines = list_all_files(rootdir)
    random.seed(time.time())
    video_indices = list(range(len(lines)))
    random.shuffle(video_indices)
    while 1:
        random.seed(time.time())
        random.shuffle(video_indices)
        for index in video_indices:
            tmp = lines[index]
            ret_arr = np.load(tmp)
            X.append(ret_arr)
            Y.append(ret_arr)
            cnt += 1
            if cnt==batch_size:
                cnt = 0
                yield  (np.array(X),np.array(Y))
                X = []
                Y = []
if __name__ == "__main__":
    base_path = "/media/gzs/denys/paper/auto_train/pedestrian2/"
#    afl = list_all_files(base_path)
#    print(len(afl))
    batch_size = 2
    generate_data_from_train(batch_size,base_path)