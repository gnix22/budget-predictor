import pandas as pd
import numpy as np
from src.time_series_predictor import TimeSeriesPredictor

df = pd.read_csv("./data/account-data-checking.csv")

df = df.loc[:,["Post Date","Credit"]]
print(df.head())

time_series = TimeSeriesPredictor(df)
outliers = time_series.get_outliers()
print(outliers)
