# coding: utf-8

# # Stock Prediction using attention-based convLSTM

# In[244]:

from __future__ import print_function, division

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import matplotlib

from keras.layers import Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers import DepthwiseConv2D, MaxPooling3D, MaxPooling1D, Conv1D, \
    RepeatVector
from keras.layers import Lambda, Dot, Multiply, Input, Permute, LSTM, Dense, \
    TimeDistributed
from keras.layers import Conv2D, ConvLSTM2D, Flatten, Reshape, \
    BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam
from keras.utils import to_categorical
import keras.backend as K

from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split




company_name = "AMZN"
directory = "../data/New/{}/".format(company_name)
output_directory = "../results/{}/conv_lstm/".format(company_name)

TIME_DIM = 4
FILE_NAME = directory + '{}_s.npy'.format(company_name)
TRAIN_SPLIT = 0.75

matplotlib.use('Agg')




# In[245]:


class SP():
    def __init__(self):
        self.rows = TIME_DIM
        self.cols = 9
        self.h_activation = 'selu'
        self.o_activation = 'selu'
        self.optimizer = Adam(lr=0.00005)

        self.model = self.build_model()
        self.model.compile(loss=self.loss,
                           optimizer=self.optimizer,
                           metrics=['mae'])


# In[246]:


class SP(SP):
    def time_dist(self, data):
        x = np.zeros((data.shape[0] - TIME_DIM + 1, TIME_DIM, data.shape[1]),
                     dtype='float')
        y = np.zeros((data.shape[0] - TIME_DIM + 1, data.shape[1]),
                     dtype='float')
        SIZE = data.shape[0] - TIME_DIM + 1
        for i in range(SIZE - 1):
            for j in range(TIME_DIM):
                x[i, j, :] = data[i + j, :]
            y[i] = data[i + j + 1]
        return x, y


# In[247]:


class SP(SP):
    def data_norm(self, x):
        if self.flag == 0:
            self.mu = np.mean(x, axis=0)
            self.variance = np.var(x, axis=0)
            self.flag = 1
        x = (x - self.mu) / self.variance
        return x


# In[248]:


class SP(SP):
    def data_loader(self):
        data = np.load(FILE_NAME)
        loc = int(TRAIN_SPLIT * data.shape[0])
        X_train = data[0:loc]
        X_test = data[loc:]
        # X_train = self.data_norm(X_train)
        # X_test = self.data_norm(X_test)
        X_train, Y_train = self.time_dist(X_train)
        X_test, Y_test = self.time_dist(X_test)
        # X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=1-TRAIN_SPLIT, random_state=42)
        return X_train, Y_train, X_test, Y_test


# In[249]:


class SP(SP):
    def loss(self, y_true, y_pred):
        return K.mean(abs(y_true - y_pred))


# In[250]:


class SP(SP):
    def build_model(self):
        # Conv LSTM
        noise = Input(shape=(self.rows,self.cols), name='Input')
        l1 = Reshape((noise.shape[1],noise.shape[2],1), name='Reshape_s1s21')(noise)
        l2 = Reshape((noise.shape[1],1,noise.shape[2]), name='Reshape_s11s2')(noise)
        aff = Lambda(lambda x: tf.matmul(x[0],x[1]), name='Affinity')([l1,l2])
        aff = Reshape((TIME_DIM,noise.shape[2],noise.shape[2],1), name='Affinity_t_sliced')(aff)
        seq1 = ConvLSTM2D(32, 3, activation = self.h_activation,return_sequences=True)(aff)
        seq2 = ConvLSTM2D(32, 3, activation = self.h_activation,return_sequences=True)(seq1)
        seq = Reshape((1,TIME_DIM,seq2.shape[2]*seq2.shape[3]*seq2.shape[4]))(seq2)
        per = Permute((1,3,2))(seq)
        wgh = DepthwiseConv2D(1,1,activation = 'softmax')(per)
        aff = Multiply()([wgh,per])
        mean = Lambda(lambda x: K.mean(x, axis=3))(aff)
        per2 = Permute((2,1))(mean)
        pool3 = Flatten()(per2)
        dense2 = Dense(self.cols, activation = self.o_activation)(pool3)
        model = Model(inputs=noise,outputs=dense2)
        model.summary()
        return model

        # # LSTM
        # inp = Input(shape=(self.rows, self.cols), name='Input')
        # ls1 = LSTM(15, return_sequences=True)(inp)
        # ls2 = LSTM(15, return_sequences=True)(ls1)
        # # ls3 = LSTM(15,return_sequences=True)(ls2)
        # pool3 = Flatten()(ls2)
        # dense3 = Dense(self.cols, activation=self.o_activation)(pool3)
        # model = Model(inputs=inp, outputs=dense3)
        # model.summary()
        # return model


# In[251]:


class SP(SP):
    def train(self, epochs, batch_size):
        X_train, Y_train, X_test, Y_test = self.data_loader()
        self.model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
                       validation_split=0.2)
        Y_pred = self.model.predict(X_train)
        mean_ = np.mean(((Y_pred - Y_train) ** 2), axis=0)
        mean_ = mean_[3]
        rms_val = np.sqrt(mean_)

        rms_val = np.reshape(rms_val, (1, 1))
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        np.savetxt(output_directory + 'pred_{0}.csv'.format(epochs), Y_pred,
                   delimiter=',')
        np.savetxt(output_directory + 'rmse_{0}.csv'.format(epochs), rms_val,
                   delimiter=',')
        plt.plot(list(range(0, Y_pred.shape[0])), Y_pred[:, 3], '-',
                 color='red')
        plt.plot(list(range(0, Y_train.shape[0])), Y_train[:, 3], '-',
                 color='blue')
        plt.savefig(output_directory + 'plot_{0}.png'.format(epochs))
        plt.close()


# In[252]:


if __name__ == '__main__':
    cnnLSTM = SP()
    cnnLSTM.train(epochs=100, batch_size=128)

