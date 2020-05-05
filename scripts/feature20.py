import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# companies = ["AAPL"]
companies = ["AAPL", "AMZN", "AXP", "CSCO", "JPM", "MMM", "PG", "VZ", "WMT", "CVX"]
for company in companies:
    # Fetch First 9 features
    print("Company: " + company)
    csv_file = "../data/New/{}/{}.csv".format(company, company)
    df = pd.read_csv(csv_file)
    df = df[-200000:]

    # FFT
    df["fft"] = np.fft.fft(np.asarray(df["close"].tolist()))
    # df["real_fft"] = np.fft.rfftn(np.asarray(df["close"].tolist()))

    # PLot for FFT
    l = []
    fft_list = np.asarray(df['fft'].tolist())
    for num_ in [10, 20, 50]:
        fft_list_m10 = np.copy(fft_list)
        fft_list_m10[num_:-num_] = 0
        f = np.fft.ifft(fft_list_m10)
        l.append(f)
    df["fft10"] = l[0].real
    df["fft20"] = l[1].real
    df["fft50"] = l[2].real
    # x_axis = [i for i in range(len(df))]
    # plt.plot(x_axis, df["close"], x_axis, l[0], x_axis, l[1], x_axis, l[2])
    # plt.legend(("Original", "10 components", "20 components", "50 components"))
    # plt.show()


    # Moving Average
    df['ma480'] = df['close'].rolling(window=480).mean()
    df['ma1440'] = df['close'].rolling(window=1440).mean()
    df['ma3360'] = df['close'].rolling(window=3360).mean()


    # MACD
    df['26ema'] = pd.DataFrame.ewm(df['close'], span=26).mean()
    df['12ema'] = pd.DataFrame.ewm(df['close'], span=12).mean()
    df['macd'] = (df['12ema'] - df['26ema'])

    # Bollinger Bands
    df['20ma'] = df['close'].rolling(window=20).mean()
    df['20sd'] = df['20ma'].std()
    df['upper_band'] = df['20ma'] + (df['20sd'] * 2)
    df['lower_band'] = df['20ma'] - (df['20sd'] * 2)

    # Exponential Moving Avg
    df['ema50'] = pd.DataFrame.ewm(df['close'], com=0.5).mean()
    df['ema25'] = pd.DataFrame.ewm(df['close'], com=0.25).mean()

    # Plot the graph comparing ground truth with predictions
    # x_axis = [i for i in range(len(df))]
    # # plt.plot(x_axis, df["close"], x_axis, df["ema50"], x_axis, df["ema25"])
    # # plt.legend(("Original", "EMA 0.5", "EMA 0.25"))
    # plt.plot(x_axis, df["close"], x_axis, df["ema50"])
    # plt.legend(("Original", "EMA 0.5"))
    # plt.show()


    # Dropping unnecessary columns
    df = df.drop(columns=["fft", "26ema", "12ema", "20ma", "20sd"])
    df.dropna(inplace=True)

    df = df.set_index("date")
    df.to_csv("../data/New/{}/{}_20.csv".format(company, company))
    np.save("../data/New/{}/{}_20".format(company, company), np.array(df))
