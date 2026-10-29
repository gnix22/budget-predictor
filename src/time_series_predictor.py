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
    # for outputs, might want to add a verbose functionality eventually ...............

    # find outliers using modified z-score and winsorize found outliers to preserve data
    def _winsorize_outliers(self):
        # drop date column and assign remaining series data for outlier detection
        df = self.df.copy()
        drop_cols = df.loc[:, ~df.columns.str.contains("Date", case=False)]
        time_series_col_name = drop_cols.columns[0]
        # removes "" that show up in series
        time_series_data = pd.Series(pd.to_numeric(drop_cols[time_series_col_name], errors='coerce')).dropna()
        median = np.median(time_series_data)
        mad = np.median(np.abs(time_series_data - median))

        # Get the first Date column name as a string
        date_col_name = df.columns[df.columns.str.contains("Date", case=False)][0]

        scoring_df = pd.DataFrame({
            time_series_col_name: time_series_data,
            "mod_z_score": 0.6745 * (time_series_data - median) / mad,
            # Slice using the same index as time_series_data to keep sizes identical
            "Date": df.loc[time_series_data.index, date_col_name]
        })
        outliers_mad = scoring_df[np.abs(scoring_df["mod_z_score"]) > 3.5]

        #print("Outliers found via MAD:\n", outliers_mad[[time_series_col_name]])

        # now clip outliers to keep hold of data
        threshold = 3.5

        deviation_limit = (threshold * mad) / 0.6745
        lower_bound = median - deviation_limit
        upper_bound = median + deviation_limit

        #print(f"Credit lower cutoff boundary: {lower_bound:.2f}")
        #print(f"Credit upper cutoff boundary: {upper_bound:.2f}")

        scoring_df[time_series_col_name] = scoring_df[time_series_col_name].clip(lower=lower_bound, upper=upper_bound)
        return scoring_df

    def _prep_data(self):
        # clip data and get month sums for target column
        df = self._winsorize_outliers()
        target_col = str(df.columns[~df.columns.str.contains("Date|mod_z_score", case=False)][0])
        df["Date"] = pd.to_datetime(df["Date"])
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        month_sums = df.groupby(['Year', 'Month'], as_index=False)[target_col].sum()
        # data split 80-20
        split_idx = int(len(month_sums) * 0.8)
        data = np.asarray(month_sums[target_col])
        train = data[:split_idx]
        test = data[split_idx:]
        return train, test

    def model_fit(self):
        print("Starting ARIMA Training")
        # get auto ordering from auto_arima, then use for getting best model
        def _auto_order(training_data):
            order = auto_arima(training_data).order
            #print("best ordering for given data: ", order)
            return order

        train, test = self._prep_data()
        order = _auto_order(train)
        model = ARIMA(train, order = order)
        res = model.fit()
        #print(res.summary())
        return res

    def get_forecast(self, result):
        return result.get_forecast(steps = 1).predicted_mean

