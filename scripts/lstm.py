import math
import os

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, ConvLSTM2D, Flatten, BatchNormalization, LSTM
from keras.optimizers import adam

import matplotlib.pyplot as plt
from matplotlib import pylab

from api.alphavantage import AlphaVantage
from util import split_df, CSVFile, plot_graph

output_folder = "lstm_stock_hourly_lf2"
directory = "../data/" + output_folder
output_filename = directory + "/stock_output.csv"


def data_loader():
    data = np.load(FILE_NAME)
    loc = int(TRAIN_SPLIT * data.shape[0])
    X_train = data[0:loc]
    X_test = data[loc:]
    # X_train = self.data_norm(X_train)
    # X_test = self.data_norm(X_test)
    X_train, Y_train = time_dist(X_train)
    X_test, Y_test = time_dist(X_test)
    return X_train, Y_train, X_test, Y_test

def data_norm(x):
    if flag == 0:
        mu = np.mean(x, axis=0)
        variance = np.var(x, axis=0)
        flag = 1
    x = (x - mu) / variance
    return x


def time_dist(data):
    x = np.zeros((data.shape[0] - TIME_DIM + 1, TIME_DIM, data.shape[1]),
                 dtype='float')
    y = np.zeros((data.shape[0] - TIME_DIM + 1, data.shape[1]), dtype='float')
    SIZE = data.shape[0] - TIME_DIM + 1
    for i in range(SIZE - 1):
        for j in range(TIME_DIM):
            x[i, j, :] = data[i + j, :]
        y[i] = data[i + j + 1]
    return x, y


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
    directory = "../io/output1/lstm/LookForward_{}/LookBack_{}/"\
        .format(str(look_forward), str(look_back))

    csv_directory = directory

    input_directory = "../io/input/"

    regressor_name = "LSTM"

    # if directory doesn't exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # if directory for CSV doesn't exist, create it
    if not os.path.exists(csv_directory):
        os.makedirs(csv_directory)

    # Create CSV File for keeping track of RMSE
    csv_file = CSVFile(csv_directory + "rmse.csv", headers=["Company Name",
                                                            "Regressor Name",
                                                            "RMSE", "MAE"])

    # For each company in the DJIA
    for company_name, stock_ticker in company_info.items():
        print("Company: " + company_name)

        stock_details_df = pd.read_csv(input_directory +
                                       "{}_20.csv".format(stock_ticker))

        stock_details_df = stock_details_df.set_index("date")
        no_of_features = len(stock_details_df.columns)
        stock_details_df = normalize_every_columns(stock_details_df)
        stock_details_df = stock_details_df.dropna()
        train_df, predict_df = split_df(stock_details_df, 0.2)
        X_train, y_train = convert_to_nn_input(train_df, look_back=look_back,
                                               look_forward=look_forward)
        X_test, y_test = convert_to_nn_input(predict_df, look_back=look_back,
                                             look_forward=look_forward)

        model = Sequential()
        model.add(LSTM(10, input_shape=(look_back * no_of_features, 1)))
        model.add(Dense(1, activation='relu'))
        model.compile(optimizer='adam', loss='mse')

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        history = model.fit(X_train, y_train, epochs=100,
                            validation_data=(X_test, y_test),
                            shuffle=False)
        y_test_predictions = model.predict(X_test)

        # Calculate the RMSE and MAE between ground truth and predictions and
        # add to the CSV file
        rmse_val = mean_squared_error(y_test, y_test_predictions) ** 0.5
        mae_val = mean_absolute_error(y_test, y_test_predictions)
        csv_file.add_row([company_name, regressor_name,
                          str(rmse_val), str(mae_val)])

        # Plot the graph comparing ground truth with predictions
        plot_graph(y_test, y_test_predictions,
                   [i for i in range(len(y_test))],
                   directory + company_name + "_" + regressor_name + ".png")
