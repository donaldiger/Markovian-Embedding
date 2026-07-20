import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

final_data = adjust_outliers(clean_data(load_data("/Users/mathisvernier/Markovian-Embedding/data/raw/2025-09-fred-md.csv")[0]))

def write_new_data(X, path = "/Users/mathisvernier/Markovian-Embedding/data/processed/processed_FRED_MD.csv"):
    X.to_csv(path)

write_new_data(final_data)

"""
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].bar(range(1, 21), evr[:20], color="steelblue"); ax[0].set_title("explained variance per PC")
ax[1].plot(range(1, 21), np.cumsum(evr[:20]), "o-", ms=3, color="darkred")
ax[1].axhline(.5, ls=":", c="gray"); ax[1].set_title("cumulative")
for a in ax: a.set_xlabel("component"); a.tick_params(labelsize=8)
fig.tight_layout(); fig.savefig("fig3_scree.png", dpi=130); plt.close(fig)
"""