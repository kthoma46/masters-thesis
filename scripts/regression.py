import os
import numpy as np
import pandas as pd
import time

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, HuberRegressor, Ridge
from sklearn.svm import SVR
from util import split_df, plot_graph, CSVFile


def fetch_X_y(df: pd.DataFrame):
    X = np.array(df.drop("label", 1))
    # X = preprocessing(X)
    y = np.array(df["label"])
    return X,y


if __name__ == '__main__':

    # Variables that can be changed
    look_forward = 3000
    label_target = "close"
    # Dow-Jones Index Companies
    company_info = {"3M": "MMM", "Amazon": "AMZN", "American Express": "AXP",
                    "Apple": "AAPL", "Chevron": "CVX", "Cisco Systems": "CSCO",
                    "JPMorgan Chase": "JPM", "Procter & Gamble": "PG",
                    "Verizon": "VZ", "Walmart": "WMT"}
    # company_info = {"3M": "MMM"}
    # regressor = LinearRegression()
    # regressor = HuberRegressor()
    regressor = Ridge()
    # directory = "../io/output2/regression/LookForward_{}/"\
    #     .format(str(look_forward))
    # csv_directory = directory
    regressor_name = str(regressor).split("(")[0]
    # directory += regressor_name + "/"

    input_directory = "../io/input/"
    time_csv_directory = "../io/time/"

    # if directory doesn't exist, create it
    # if not os.path.exists(directory):
    #     os.makedirs(directory)

    # if directory for CSV doesn't exist, create it
    # if not os.path.exists(csv_directory):
    #         os.makedirs(csv_directory)

    # Create CSV File for keeping track of RMSE
    # csv_file = CSVFile(csv_directory + "rmse.csv", headers=["Company Name",
    #                                                         "Regressor Name",
    #                                                         "RMSE", "MAE"])

    # if directory for CSV doesn't exist, create it
    if not os.path.exists(time_csv_directory):
            os.makedirs(time_csv_directory)

    time_csv_file = CSVFile(time_csv_directory + "time_{}.csv".format(str(look_forward)),
                            headers=["Company Name", "Regressor Name",
                                     "Total Time to Train",
                                     "Time taken to test 1 sample"])


    avg_train = []
    avg_test = []
    for company_name, stock_ticker in company_info.items():
        # print("Company: " + company_name)

        stock_details_df = pd.read_csv(input_directory +
                                       "{}_20.csv".format(stock_ticker))

        stock_details_df = stock_details_df.set_index("date")
        stock_details_df["label"] = stock_details_df[label_target].\
            shift(-look_forward)
        stock_details_df.dropna(inplace=True)

        # Split the time series data 80-20 split
        stock_details_df_train, stock_details_df_test = \
            split_df(stock_details_df, 0.8)

        # Fetch training and testing data for the model. No need for
        # validation data
        X_train, y_train = fetch_X_y(stock_details_df_train)
        X_test, y_test = fetch_X_y(stock_details_df_test)

        # Train the model
        training_start_time = time.time()
        regressor.fit(X_train, y_train)
        training_end_time = time.time()

        # Predict values
        testing_start_time = time.time()
        y_test_predictions = regressor.predict(X_test)
        testing_end_time = time.time()
        test_samples_no = len(X_test)

        # Calculate the RMSE and MAE between ground truth and predictions and
        # add to the CSV file
        # rmse_val = mean_squared_error(y_test, y_test_predictions) ** 0.5
        # mae_val = mean_absolute_error(y_test, y_test_predictions)
        # csv_file.add_row([company_name, regressor_name,
        #                   str(rmse_val), str(mae_val)])

        # Plot the graph comparing ground truth with predictions
        # plot_graph(y_test, y_test_predictions,
        #            [i for i in range(len(y_test))],
        #            directory + company_name + "_" + regressor_name + ".png")

        train_time = training_end_time - training_start_time
        test_time = testing_end_time - testing_start_time
        sample_test_time = test_time/test_samples_no
        # print("Time to train: " + str(train_time))
        # print("Time to test: " + str(test_time))
        # print("Time to test for 1 sample: " + str(test_time/test_samples_no))
        time_csv_file.add_row([company_name, regressor_name, train_time,
                               sample_test_time])
        avg_train.append(train_time)
        avg_test.append(sample_test_time)
    time_csv_file.add_row(["Average", regressor_name, sum(avg_train)/10.0,
                           sum(avg_test)/10.0])