import pandas as pd
import glob
import os
import numpy as np

companies = ["AMZN", "AXP", "CSCO", "JPM", "MMM", "PG", "VZ", "WMT", "CVX"]
for company in companies:
    print("Company: " + company)
    pathA1 = "../data/New/{}/".format(company)

    from pathlib import Path

    pathlist = Path(pathA1).glob('**/*.csv')
    col = ["date", "open", "high", "low", "close", "volume", "no_of_trades",
           "weighted_avg_price"]

    global_df = None
    for path in pathlist:
        # because path is object not string
        path_in_str = str(path)
        df = pd.read_csv(path_in_str, header=None)
        if global_df is None:
            global_df = df
        else:
            global_df = pd.concat([global_df, df])

    global_df.columns = col
    global_df = global_df.set_index('date')
    global_df['volatility'] = (global_df['high'] - global_df['low']) / global_df['low'] * 100.0
    global_df['percent_change'] = (global_df['close'] - global_df['open']) / global_df['open'] * 100.0
    global_df.to_csv(pathA1 + "{}.csv".format(company))
    array = np.array(global_df)
    np.save(pathA1 + "{}_numpy".format(company), array)
print("Done!")
