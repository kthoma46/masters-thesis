# coding: utf-8

# # Stock Prediction using attention-based convLSTM

# In[244]:


from __future__ import print_function, division

import os

from util import CSVFile

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time

matplotlib.use('Agg')

from keras.layers import Input, LSTM, Dense
from keras.layers import Flatten
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K

avg_train = []
avg_test = []

class SP():
    def __init__(self, directory, input_file, csv_file, look_back, look_forward, train_split, o_act):
        self.directory = directory
        self.look_back = look_back
        self.look_forward = look_forward
        self.input_file = input_file
        self.csv_file = csv_file
        self.train_split = train_split
        self.rows = self.look_back
        self.cols = 20
        self.h_activation = 'selu'
        self.o_activation = o_act
        self.optimizer = Adam(lr=0.005)

        self.model = self.build_model()
        self.model.compile(loss=self.loss,
                           optimizer=self.optimizer,
                           metrics=['mae'])

    def time_dist(self, data):
        x = np.zeros((data.shape[0] - self.look_back + 1, self.look_back,
                      data.shape[1]), dtype='float')
        y = np.zeros((data.shape[0] - self.look_back + 1, data.shape[1]),
                     dtype='float')
        SIZE = data.shape[0] - self.look_back + 1
        for i in range(SIZE - self.look_back - self.look_forward):
            for j in range(self.look_back):
                x[i, j, :] = data[i + j, :]
            y[i] = data[i + j + self.look_forward]
        return x, y

    def data_norm(self, x):
        if self.flag == 0:
            self.mu = np.mean(x, axis=0)
            self.variance = np.var(x, axis=0)
            self.flag = 1
        x = (x - self.mu) / self.variance
        return x

    def data_loader(self):
        data = np.load(self.input_file)
        loc = int(self.train_split * data.shape[0])
        X_train = data[0:loc]
        X_test = data[loc:]
        X_train, Y_train = self.time_dist(X_train)
        X_test, Y_test = self.time_dist(X_test)
        return X_train, Y_train, X_test, Y_test

    def loss(self, y_true, y_pred):
        return K.mean(abs(y_true - y_pred))

    def build_model(self):
        inp = Input(shape=(self.rows, self.cols), name='Input')
        ls1 = LSTM(40, return_sequences=True)(inp)
        # ls2 = LSTM(15, return_sequences=True)(ls1)
        # pool3 = Flatten()(ls2)
        pool3 = Flatten()(ls1)
        dense3 = Dense(self.cols, activation=self.o_activation)(pool3)
        dense4 = Dense(self.cols, activation=self.o_activation)(dense3)
        model = Model(inputs=inp, outputs=dense4)
        model.summary()
        return model

    def train(self, epochs, batch_size):
        X_train, Y_train, X_test, Y_test = self.data_loader()
        training_start_time = time.time()
        self.model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
                       validation_split=0.2)
        training_end_time = time.time()

        testing_start_time = time.time()
        Y_pred = self.model.predict(X_test)
        testing_end_time = time.time()
        test_samples_no = len(X_test)

        # # RMSE
        # mean_ = np.mean(((Y_pred - Y_test) ** 2), axis=0)
        # mean_ = mean_[3]
        # rms_val = np.sqrt(mean_)
        # rms_val = np.reshape(rms_val, (1, 1))
        #
        # # MAE
        # mean_ = np.mean((np.abs(Y_pred - Y_test)), axis=0)
        # mae_val = mean_[3]
        # mae_val = np.reshape(mae_val, (1, 1))
        #
        # csv_file.add_row([company_name, regressor_name, o_act,
        #                   str(rms_val), str(mae_val)])
        # plt.plot(list(range(0, Y_pred.shape[0])), Y_pred[:, 3], '-',
        #          color='red')
        # plt.plot(list(range(0, Y_train.shape[0])), Y_train[:, 3], '-',
        #          color='blue')
        # plt.savefig(self.directory + company_name + "_" + regressor_name +
        #             "{}.png".format(epochs))
        # plt.close()

        train_time = training_end_time - training_start_time
        test_time = testing_end_time - testing_start_time
        sample_test_time = test_time / test_samples_no
        csv_file.add_row([company_name, regressor_name, train_time,
                          sample_test_time])
        avg_train.append(train_time)
        avg_test.append(sample_test_time)


if __name__ == '__main__':
    # Variables that can be changed
    look_forward = 1
    look_back = 4
    label_column = "close"
    # Dow-Jones Index Companies
    company_info = {"3M": "MMM", "Amazon": "AMZN", "American Express": "AXP",
                    "Apple": "AAPL", "Chevron": "CVX", "Cisco Systems": "CSCO",
                    "JPMorgan Chase": "JPM", "Procter & Gamble": "PG",
                    "Verizon": "VZ", "Walmart": "WMT"}
    # directory = "../io/output/lstm/LookForward_{}/LookBack_{}/" \
    #     .format(str(look_forward), str(look_back))

    # csv_directory = directory
    csv_directory = "../io/time/"

    input_directory = "../io/input/"

    regressor_name = "LSTM"

    # if directory for CSV doesn't exist, create it
    if not os.path.exists(csv_directory):
        os.makedirs(csv_directory)

    # Create CSV File for keeping track of RMSE
    # csv_file = CSVFile(csv_directory + "rmse.csv", headers=["Company Name",
    #                                                         "Regressor Name",
    #                                                         "Output Activation",
    #                                                         "RMSE", "MAE"])
    csv_file = CSVFile(csv_directory + "time_{}.csv".format(str(look_forward)),
                       headers=["Company Name", "Regressor Name",
                                "Total Time to Train",
                                "Time taken to test 1 sample"])

    # o_act_list = ['selu', 'relu', 'sigmoid', 'linear']
    o_act_list = ["linear"]
    # For each company in the DJIA
    for company_name, stock_ticker in company_info.items():
        # print("Company: " + company_name)

        for o_act in o_act_list:
            # print(o_act)
            # new_directory = directory + o_act + "/"
            # if directory doesn't exist, create it
            # if not os.path.exists(new_directory):
            #     os.makedirs(new_directory)
            input_file = input_directory + "{}_20.npy".format(stock_ticker)
            cnnLSTM = SP(directory=None, input_file=input_file,
                         csv_file=csv_file, look_back=look_back,
                         look_forward=look_forward, train_split=0.75,
                         o_act=o_act)
            cnnLSTM.train(epochs=10, batch_size=32)
    csv_file.add_row(["Average", regressor_name, sum(avg_train) / 10.0,
                           sum(avg_test) / 10.0])

