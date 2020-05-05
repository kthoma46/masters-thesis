import pandas

if __name__ == '__main__':
    stock_ticker = "MMM"
    directory = "../data/static/"
    stock_details_df = pandas.read_csv(directory + stock_ticker + ".csv")
    print()
