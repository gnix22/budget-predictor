import pandas as pd
import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse

class TimeSeriesPredictor:
    """
    Uses DataFrame containing only date and time series data wanted to sum.
    That is, if you want to predict income using time series, only the post date
    and sum can be passed in as arg. This allows monthly sums to be taken.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_outliers(self):
        # drop date column and assign remaining series data for outlier detection
        df = self.df.copy()
        drop_cols = df.loc[:, ~df.columns.str.contains('Date', case=False)]
        time_series_col_name = drop_cols.columns[0]
        # removes "" that show up in series
        time_series_data = pd.to_numeric(drop_cols[time_series_col_name], errors='coerce').dropna()
        median = np.median(time_series_data)
        mad = np.median(np.abs(time_series_data - median))

        scoring_df = pd.DataFrame({
            time_series_col_name: time_series_data,
            "mod_z_score": 0.6745 * (time_series_data - median) / mad
        })
        outliers_mad = scoring_df[np.abs(scoring_df["mod_z_score"]) > 3.5]

        print("Outliers found via MAD:\n", outliers_mad[[time_series_col_name]])
    #threshold = 3.5

    #deviation_limit = (threshold * mad) / 0.6745
    #lower_bound = median - deviation_limit
    #upper_bound = median + deviation_limit

    #income["Credit_winsorized"] = income["Credit"].clip(lower=lower_bound, upper=upper_bound)

    #print(f"Credit lower cutoff boundary: {lower_bound:.2f}")
    #print(f"Credit upper cutoff boundary: {upper_bound:.2f}")
