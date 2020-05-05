import numpy as np
companies = ["AAPL"]
for company in companies:
    x = np.load('../data/New/{}/{}_numpy.npy'.format(company, company))
    x = x[-200000:,:]
    print(x.shape)
    np.save('../data/New/{}/{}_s.npy'.format(company, company), x)
