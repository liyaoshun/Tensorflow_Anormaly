# Tensorflow_Anormaly
使用深度神经网络的技术在视屏中检测异常并定位

代码和效果很快将会提交


VGG_conv3_3_ROC.png是使用的VGG16进行特这特区后在UCSD_ped2所有测试集上计算的ROC曲线。当前EER==18.3468%.

/data/Test/下的.png图像展示的是异常检测并定位的一部分较好的效果图。

当前正在采集数据训练模型和模型调参。

##########################################################

经过模型超参数的修改和神经网络的训练，EER==16.5323%  y==83.3874(使用的是我们自己构建的模型和重新在UCF101上训练得到的结构)
我们将EER从18.3468%降低到了16.5323%.

# Tips: 计算机硬盘损害，丢失了实验相关数据。
