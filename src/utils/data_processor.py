import numpy  as np
import pandas as pd

class DataProcessor:
    def __init__(self, data):
        self.data = data

    def _transform_series(self, x, code):
        if code == 1:  return x
        if code == 2:  return x.diff()
        if code == 3:  return x.diff().diff()
        if code == 4:  return np.log(x)
        if code == 5:  return np.log(x).diff()
        if code == 6:  return np.log(x).diff().diff()
        if code == 7:  return x.pct_change().diff()
        raise ValueError(f"unknown tcode {code}")

    def transform(self, df, tcodes):
        out = pd.DataFrame({c: self._transform_series(df[c], tcodes[c]) for c in df.columns})
        return out.iloc[2:]

    def clean_data(self, X, max_missing=24):
        """Drop series with too many NaNs, then drop any residual NaN rows/interp short gaps."""
        n_miss = X.isna().sum()
        dropped = n_miss[n_miss > max_missing].index.tolist()
        if dropped:
            print(f"dropping {len(dropped)} series with big gaps: {dropped}")
        X = X.drop(columns=dropped)
        X = X.interpolate(limit=3, limit_area="inside")
        return X.dropna()

    def adjust_outliers(self, X, k=10.0):
        #try to get rid of big outliers such as corona months and replace the values
        med, iqr = X.median(), X.quantile(0.75) - X.quantile(0.25)
        mask = (X - med).abs() > k * iqr
        n = int(mask.values.sum())
        print(f"outlier rule flags {n} observations "
              f"({mask.any(axis=1).sum()} months affected, incl. COVID if in sample)")
        return X.mask(mask).interpolate(limit_direction="both")

    def standardize(self, X):
        train = X.copy()
        mu, sd = train.mean(), train.std(ddof=0)
        return (X - mu) / sd, mu, sd

    def process(self, df, tcodes, standardize=True):
        """Run the full transform -> clean -> outlier-adjust (-> standardize) pipeline."""
        X = self.transform(df, tcodes)
        X = self.clean_data(X)
        X = self.adjust_outliers(X)
        if standardize:
            return self.standardize(X)
        return X
