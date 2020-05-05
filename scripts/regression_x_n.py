import os
import numpy as np
import pandas as pd

from api.alphavantage import AlphaVantage
from sklearn.linear_model import LinearRegression, HuberRegressor, Ridge
from sklearn.svm import SVR
from util import split_df, plot_graph, rmse, CSVFile


def fetch_X(df, lookback=4) -> np.ndarray:
    length = len(df) - lookback - 1
    X = None
    X_df = df
    i = 0
    while (i + lookback + 1) != len(df):
        X_df_portion = X_df[i:i + lookback]
        if X is None:
            X = np.array(X_df_portion)
        else:
            X = np.append(X, X_df_portion)

        i += 1
    X = np.reshape(X, (length, lookback * no_of_features))
    return X


def fetch_y(df, labeltarget, lookback=4, lookforward=1) -> np.ndarray:
    length = len(df) - lookback - 1
    y = None
    y_df = df[labeltarget]
    i = 0
    while (i + lookback + 1) != len(df):
        y_df_portion = y_df[i + lookback:i + lookback + lookforward]
        if y is None:
            y = np.array(y_df_portion)
        else:
            y = np.append(y, y_df_portion)

        i += 1
    y = np.reshape(y, (length, lookforward))
    return y


if __name__ == '__main__':
    # Variables that can be changed
    label = "close"
    look_back = 4
    limit = 30
    look_forward = 1
    company_info = {"Apple": "AAPL"}
    # regressor = LinearRegression()
    regressor = HuberRegressor()
    # regressor = Ridge()
    # regressor = SVR()
    directory = "../data/regression/"
    if len(company_info) == 30:
        company_names = "TEST/"
    else:
        company_names = ",".join([key for key in company_info.keys()])
    directory += company_names + "/"
    csv_directory = directory
    regressor_name = str(regressor).split("(")[0]
    directory += regressor_name + "/"

    # if directory doesn't exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # if directory for CSV doesn't exist, create it
    if not os.path.exists(csv_directory):
        os.makedirs(csv_directory)

    # Create CSV File for keeping track of RMSE
    csv_file = CSVFile(csv_directory + "rmse.csv", headers=["Company Name",
                                                            "Regressor Name",
                                                            "Test Data Size",
                                                            "Label"
                                                            "RMSE"])

    # Stock Data API
    alphavantage = AlphaVantage()
    for company_name, stock_ticker in company_info.items():
        print("Company: " + company_name)
        # Fetch stock data for particular company
        stock_details_df = alphavantage. \
            fetch_daily_stock_details_as_df(stock_ticker)

        no_of_features = len(stock_details_df.columns)

        # Split the time series data 80-20 split
        stock_details_df_train, stock_details_df_test = \
            split_df(stock_details_df, 0.8)

        X_test_new_row = []
        X_train = fetch_X(stock_details_df_train, look_back)
        X_test = fetch_X(stock_details_df_test, look_back)[:1]
        i = 0
        num_list = [j for j in range(no_of_features)]
        y_test = fetch_y(stock_details_df_test, label, look_back)
        y_test_predictions = []
        for i in range(len(y_test)):
            X_test_new_row = np.delete(X_test, num_list, axis=1)
            for label_target in stock_details_df.columns:
                # Fetch training and testing data for the model. No need for
                # validation data
                y_train = fetch_y(stock_details_df_train, label_target,
                                  look_back)


                # Train the model
                regressor.fit(X_train, y_train.ravel())

                # Predict values
                predicted_y_value = regressor.predict(X_test)
                if label_target == label:
                    y_test_predictions.append(predicted_y_value)
                X_test_new_row = np.append(X_test_new_row, predicted_y_value)

            X_test_new_row = X_test_new_row.reshape(1, no_of_features*look_back)
            X_test = np.concatenate((X_test, X_test_new_row), axis=0)
            X_test = np.delete(X_test, 0, axis=0)



        # Calculate the RMSE between ground truth and predictions and
        # add to the CSV file
        rmse_val = rmse(y_test, np.array(y_test_predictions))
        csv_file.add_row([company_name, regressor_name, str(len(y_test)),
                          label, str(rmse_val)])

        # Plot the graph comparing ground truth with predictions
        plot_graph(y_test, y_test_predictions,
                   [i for i in range(len(y_test))], directory + company_name
                   + "_" + regressor_name + "_" + label + ".png")

