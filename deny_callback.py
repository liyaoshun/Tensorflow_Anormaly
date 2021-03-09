#from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
#from keras.callbacks import EarlyStopping
#from keras.callbacks import LearningRateScheduler
#from keras.callbacks import ReduceLROnPlateau
#import keras.backend as K
import numpy as np
#import math
#import time
#import os
import matplotlib.pyplot as plt
def history():
    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
            self.losses_val = []
            self.accs = []
            self.accs_val = []

        # def on_batch_end(self, batch, logs={}):
        #     self.batch_losses.append(logs.get('loss'))
        #     self.batch_accs.append(logs.get('ACCLoss'))
        def on_epoch_end(self, epoch, logs={}):
            self.losses.append(logs.get('loss'))
            self.losses_val.append(logs.get('val_loss'))
            self.accs.append(logs.get('acc'))
            self.accs_val.append(logs.get('val_acc'))
    return LossHistory()
def save(HISTORY):
    np.save("./models/loss/loss.npy",HISTORY.losses)
    np.save("./models/loss/accs.npy",HISTORY.accs)
    np.save("./models/loss/accs_val.npy",HISTORY.accs_val)
    np.save("./models/loss/losses_val.npy",HISTORY.losses_val)
def show():
    acc = np.load("./models/loss/accs.npy")

    loss = np.load("./models/loss/loss.npy")

    acc_val = np.load("./models/loss/accs_val.npy")

    loss_val = np.load("./models/loss/losses_val.npy")

    plt.subplot(221)
    plt.plot(np.double(loss),'g-')
    plt.ylabel('loss')
    plt.xlabel('epoch')


    plt.subplot(222)
    plt.plot(np.double(acc),'g-')
    plt.ylabel('acc')
    plt.xlabel('epoch')


    plt.subplot(223)
    plt.plot(np.double(loss_val),'g-')
    plt.ylabel('loss_val')
    plt.xlabel('epoch')


    plt.subplot(224)
    plt.plot(np.double(acc_val),'g-')
    plt.ylabel('acc_val')
    plt.xlabel('epoch')
    plt.show()
if __name__ == '__main__':
    show()