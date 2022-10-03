import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import yfinance
import datetime
import numpy as np
import math
from scipy import stats

start = datetime.datetime(2000, 1, 1)
end = datetime.datetime(2022, 8, 26)
data = yfinance.download(['DX-Y.NYB', 'USDCLP=X', 'USDCOP=X','USDBRL=X', 'USDMXN=X' ],start=start, end=end,)


data = data.Close.dropna()

data_mean = data.rolling("30D").mean()
data_std = data.rolling("30D").std() * 3
data = data.where(data > data_mean - data_std)
data = data.dropna()

name = 'currencies3'
data.to_csv(f'datasets/Original_Data/{name}.csv', index=False)
