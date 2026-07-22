import numpy as np
import pandas as pd


##data loading
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

def _transform_series(x, code):
    if code == 1:  return x
    if code == 2:  return x.diff()
    if code == 3:  return x.diff().diff()
    if code == 4:  return np.log(x)
    if code == 5:  return np.log(x).diff()
    if code == 6:  return np.log(x).diff().diff()
    if code == 7:  return x.pct_change().diff()
    raise ValueError(f"unknown tcode {code}")

def transform(df, tcodes):
    out = pd.DataFrame({c: _transform_series(df[c], tcodes[c]) for c in df.columns})
    return out.iloc[2:]   

def clean_data(X, max_missing = 24):
    """Drop series with too many NaNs, then drop any residual NaN rows/interp short gaps."""
    n_miss = X.isna().sum()
    dropped = n_miss[n_miss > max_missing].index.tolist()
    if dropped:
        print(f"dropping {len(dropped)} series with big gaps: {dropped}")
    X = X.drop(columns=dropped)
    return X.dropna()

#data_new = clean_data(load_data("/Users/mathisvernier/Markovian-Embedding/data/raw/2025-09-fred-md.csv")[0])
#print(data_new)
#print(type(data_new))
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
 
def standardize(X):
    train = X.copy()
    mu, sd = train.mean(), train.std(ddof=0)
    return (X - mu) / sd, mu, sd

#final_data = adjust_outliers(clean_data(load_data("/Users/mathisvernier/Markovian-Embedding/data/raw/2025-09-fred-md.csv")[0]))

def write_new_data(X, path = "/Users/mathisvernier/Markovian-Embedding/data/processed/processed_FRED_MD.csv"):
    X.to_csv(path)

#write_new_data(final_data)
data, tcodes = load_data("/Users/mathisvernier/Markovian-Embedding/data/raw/2025-09-fred-md.csv")
X = transform(data, tcodes)
X = clean_data(X)
X = adjust_outliers(X)
X_std, mu, sd = standardize(X)

write_new_data(X_std)



##plots for qualitative data description

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

 
df, tc = load_data("/Users/mathisvernier/Markovian-Embedding/data/raw/2025-09-fred-md.csv")
level = df["UNRATE"]
diff  = level.diff()
 
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
 
#Panel 1: full-history level, April 2020 marked
ax = axes[0]
ax.plot(level.index, level, color="steelblue", lw=1)
ax.axvline(pd.Timestamp("2020-04-01"), color="darkred", ls="--", lw=1)
ax.annotate("Apr 2020", xy=(pd.Timestamp("2020-04-01"), level.max()),
            xytext=(pd.Timestamp("2010-01-01"), level.max()-1),
            fontsize=9, color="darkred")
ax.set_title("UNRATE — level, 1959–2025")
ax.set_ylabel("unemployment rate (%)")
 
#Panel 2: month-over-month change, zoomed 2018-2022
ax = axes[1]
win = diff.loc["2018-01-01":"2022-12-01"]
colors = ["darkred" if d == win.max() else "steelblue" for d in win]
ax.bar(win.index, win, width=20, color=colors)
ax.axhline(0, color="black", lw=.6)
ax.set_title("UNRATE — month-over-month change (tcode 2), 2018–2022")
ax.set_ylabel("Δ percentage points")
 
for a in axes:
    a.tick_params(labelsize=8)
 
fig.tight_layout()
fig.savefig("fig_unrate_covid.png", dpi=140)
print("saved. max jump =", diff.max(), "on", diff.idxmax().date())