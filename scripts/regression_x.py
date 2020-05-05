import os
import numpy as np
import pandas as pd

from api.alphavantage import AlphaVantage
from sklearn.linear_model import LinearRegression, HuberRegressor, Ridge
from sklearn.svm import SVR
from util import split_df, plot_graph, rmse, CSVFile


def fetch_X_y(df, look_back=4, look_forward=1):
    length = len(df) - look_back - 1
    X = None
    y = None
    X_df = df
    y_df = df[label_target]
    i = 0
    while (i + look_back + 1) != len(df):
        X_df_portion = X_df[i:i + look_back]
        if X is None:
            X = np.array(X_df_portion)
        else:
            X = np.append(X, X_df_portion)

        y_df_portion = y_df[i + look_back:i + look_back + look_forward]
        if y is None:
            y = np.array(y_df_portion)
        else:
            y = np.append(y, y_df_portion)

        i += 1
    X = np.reshape(X, (length, look_back * no_of_features))
    y = np.reshape(y, (length, look_forward))
    return X, y


if __name__ == '__main__':

    # Variables that can be changed
    look_forward = 1
    company_info = {"3M": "MMM"}
    # regressor = LinearRegression()
    regressor = HuberRegressor()
    # regressor = Ridge()
    # regressor = SVR()
    directory = "../data/data_x/"
    if len(company_info) == 30:
        company_names = "DJIA/"
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

        # Fetch stock data for particular company
        stock_details_df = alphavantage. \
            fetch_hourly_stock_details_as_df(stock_ticker)

        no_of_features = len(stock_details_df.columns)

        # Split the time series data 80-20 split
        stock_details_df_train, stock_details_df_test = \
            split_df(stock_details_df, 0.8)

        for label_target in stock_details_df.columns:
            # Fetch training and testing data for the model. No need for
            # validation data
            X_train, y_train = fetch_X_y(stock_details_df_train)
            X_test, y_test = fetch_X_y(stock_details_df_test)

            # Train the model
            regressor.fit(X_train, y_train)

            # Predict values
            y_test_predictions = regressor.predict(X_test)

            # Calculate the RMSE between ground truth and predictions and
            # add to the CSV file
            rmse_val = rmse(y_test, y_test_predictions)
            csv_file.add_row([company_name, regressor_name, str(len(y_test)),
                              label_target, str(rmse_val)])

            # Plot the graph comparing ground truth with predictions
            plot_graph(y_test, y_test_predictions,
                       [i for i in range(len(y_test))],
                       directory + company_name + "_" + regressor_name + "_"
                       + label_target + ".png")
