import os

import pandas as pd
import numpy as np

from api.alphavantage import AlphaVantage


def data_saver():
    directory = "../data/static/"
    # if directory doesn't exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    company_info = {"3M": "MMM", "American Express": "AXP",
                    "Apple": "AAPL", "Boeing": "BA",
                    "Caterpillar": "CAT", "Chevron": "CVX",
                    "Cisco Systems": "CSCO", "Coca-Cola": "KO",
                    "DowDuPont": "DWDP", "ExxonMobil": "XOM",
                    "Goldman Sachs": "GS", "The Home Depot": "HD",
                    "IBM": "IBM", "Intel": "INTC",
                    "Johnson & Johnson": "JNJ", "JPMorgan Chase": "JPM",
                    "McDonald's": "MCD", "Merck & Company": "MRK",
                    "Microsoft": "MSFT", "Nike": "NKE", "Pfizer": "PFE",
                    "Procter & Gamble": "PG", "Travelers": "TRV",
                    "UnitedHealth Group": "UNH",
                    "United Technologies": "UTX", "Verizon": "VZ",
                    "Visa": "V", "Walmart": "WMT",
                    "Walgreens Boots Alliance": "WBA", "Walt Disney": "DIS"}
    label_target = "close"
    look_forward = 1
    alphavantage = AlphaVantage()
    for company_name, stock_ticker in company_info.items():
        stock_details_df = alphavantage. \
            fetch_hourly_stock_details_as_df(stock_ticker)
        stock_details_df["label"] = stock_details_df[label_target]. \
            shift(-look_forward)
        stock_details_df.dropna(inplace=True)
        stock_details_df, label_max = normalize(stock_details_df)

        filename = directory + stock_ticker + ".csv"

        # TODO: Choose either option 1 or 2. Both won't work at the same time

        # Option 1
        stock_details_df.to_csv(filename)

        # Option 2
        array = np.array(stock_details_df)
        # TODO: Reshape if necessary
        np.savetxt(filename, array, delimiter=",")


def data_loader():
    directory = "../data/static/"
    company_info = {"3M": "MMM", "American Express": "AXP",
                    "Apple": "AAPL", "Boeing": "BA",
                    "Caterpillar": "CAT", "Chevron": "CVX",
                    "Cisco Systems": "CSCO", "Coca-Cola": "KO",
                    "DowDuPont": "DWDP", "ExxonMobil": "XOM",
                    "Goldman Sachs": "GS", "The Home Depot": "HD",
                    "IBM": "IBM", "Intel": "INTC",
                    "Johnson & Johnson": "JNJ", "JPMorgan Chase": "JPM",
                    "McDonald's": "MCD", "Merck & Company": "MRK",
                    "Microsoft": "MSFT", "Nike": "NKE", "Pfizer": "PFE",
                    "Procter & Gamble": "PG", "Travelers": "TRV",
                    "UnitedHealth Group": "UNH",
                    "United Technologies": "UTX", "Verizon": "VZ",
                    "Visa": "V", "Walmart": "WMT",
                    "Walgreens Boots Alliance": "WBA", "Walt Disney": "DIS"}
    for company_name, stock_ticker in company_info.items():
        filename = directory + stock_ticker + ".csv"

        # TODO: Choose either option 1 or 2. Both won't work at the same time

        # Option 1
        stock_details_df = pd.read_csv(filename)

        # Option 2
        array = np.genfromtxt(filename, delimiter=",")


TIME_DIM = 4


def time_distribution(data):
    x = np.zeros((data.shape[0] - TIME_DIM + 1, TIME_DIM, data.shape[1]),
                 dtype='float')
    y = np.zeros((data.shape[0] - TIME_DIM + 1, 1, data.shape[1]),
                 dtype='float')
    lab = data
    SIZE = data.shape[0] - TIME_DIM
    for i in range(SIZE):
        j = 0
        for j in range(TIME_DIM):
            x[i, j, :] = data[i + j, :]
        y[i] = lab[i + j + 1]
    return x, y

def data_manipulator():
    directory = "../data/static/"
    # company_info = {"3M": "MMM", "American Express": "AXP",
    #                 "Apple": "AAPL", "Boeing": "BA",
    #                 "Caterpillar": "CAT", "Chevron": "CVX",
    #                 "Cisco Systems": "CSCO", "Coca-Cola": "KO",
    #                 "DowDuPont": "DWDP", "ExxonMobil": "XOM",
    #                 "Goldman Sachs": "GS", "The Home Depot": "HD",
    #                 "IBM": "IBM", "Intel": "INTC",
    #                 "Johnson & Johnson": "JNJ", "JPMorgan Chase": "JPM",
    #                 "McDonald's": "MCD", "Merck & Company": "MRK",
    #                 "Microsoft": "MSFT", "Nike": "NKE", "Pfizer": "PFE",
    #                 "Procter & Gamble": "PG", "Travelers": "TRV",
    #                 "UnitedHealth Group": "UNH",
    #                 "United Technologies": "UTX", "Verizon": "VZ",
    #                 "Visa": "V", "Walmart": "WMT",
    #                 "Walgreens Boots Alliance": "WBA", "Walt Disney": "DIS"}
    company_info = {"Apple": "AAPL"}
    alphavantage = AlphaVantage()
    merged_array = None
    company_index_list = []
    company_index_list.append(0)
    for company_name, stock_ticker in company_info.items():
        print("Company: " + company_name)
        stock_details_df = alphavantage. \
            fetch_daily_stock_details_as_df(stock_ticker)
        stock_details_df.dropna(inplace=True)
        if merged_array is None:
            merged_array = np.array(stock_details_df)
            company_index_list.append(len(merged_array))
        else:
            array = np.array(stock_details_df)
            merged_array = np.concatenate((merged_array,
                                           array), axis=0)
            company_index_list.append(company_index_list[-1] + len(array))



    mu = np.mean(merged_array, axis=0)

    print("Mean: " + str(mu))
    variance = np.var(merged_array, axis=0)
    print("Var: " + str(variance))
    merged_array = (merged_array - mu) / variance
    np.save(directory + "mu", mu)
    np.save(directory + "variance", variance)


    # X, y = time_distribution(merged_array)
    # np.save(directory + "X", X)
    # np.save(directory + "y", y)

    merged_X = None
    merged_y = None
    for i in range(len(company_index_list)-1):
        X, y = time_distribution(merged_array[company_index_list[i]:
                                              company_index_list[i+1]])
        print("X size = ",X.shape)
        print("y size = ", y.shape)
        merged_X = X if merged_X is None else np.concatenate((merged_X, X),
                                                             axis=0)
        merged_y = y if merged_y is None else np.concatenate((merged_y, y),
                                                             axis=0)

    np.save(directory + "X", merged_X)
    np.save(directory + "y", merged_y)


def func():
    a = AlphaVantage()
    df = a.fetch_daily_stock_details_as_df("AAPL")
    df.to_csv("../data/AAPL.csv")


if __name__ == '__main__':
    func()