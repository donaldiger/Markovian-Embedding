import numpy as np
import pandas as pd

def load_data(path):
    df = pd.read_csv(path, sep=";")
    tcodes = df.iloc[0, 1:].astype(float).astype(int) #extract the codes of the data
    data = df.iloc[1:].copy()
    data["sasdate"] = pd.to_datetime(data["sasdate"], format="%m/%d/%Y")
    data = data.set_index("sasdate").apply(pd.to_numeric, errors="coerce") #data itself
    return data, tcodes

#data, tcodes = load_data("data/raw/2025-09-fred-md.csv")
#print(data)
#print(tcodes)

def clean_data(X, max_missing = 24):
    """Drop series with too many NaNs, then drop any residual NaN rows/interp short gaps."""
    n_miss = X.isna().sum()
    dropped = n_miss[n_miss > max_missing].index.tolist()
    if dropped:
        print(f"dropping {len(dropped)} series with big gaps: {dropped}")
    X = X.drop(columns=dropped)
    return X.dropna()

data_new = clean_data(load_data("/Users/mathisvernier/Markovian-Embedding/data/raw/2025-09-fred-md.csv")[0])
print(data_new)
print(type(data_new))
#data_new.median()

"""
df = pd.DataFrame([1,4,3])
print(type(df) == type(data_new))
df.median()
data_new.median()
"""


def adjust_outliers(X, k= 10.0):
    #try to get rid of big outliers such as corona months and replace the values
    med, iqr = X.median(), X.quantile(0.75) - X.quantile(0.25)
    mask = (X - med).abs() > k * iqr
    n = int(mask.values.sum())
    print(f"outlier rule flags {n} observations "
          f"({mask.any(axis=1).sum()} months affected, incl. COVID if in sample)")
    return X.mask(mask).interpolate(limit_direction="both")
 
def standardize_train_only(X):
    train = X.copy()
    mu, sd = train.mean(), train.std(ddof=0)
    return (X - mu) / sd, mu, sd

standardize_train_only(adjust_outliers(clean_data(load_data("/Users/mathisvernier/Markovian-Embedding/data/raw/2025-09-fred-md.csv")[0]))
)