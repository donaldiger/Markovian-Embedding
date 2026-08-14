import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import DataLoader, DataProcessor

RAW_PATH = "/Users/mathisvernier/Markovian-Embedding/data/raw/2025-09-fred-md.csv"
PROCESSED_PATH = "/Users/mathisvernier/Markovian-Embedding/data/processed/processed_FRED_MD.csv"
NS_PATH = "/Users/mathisvernier/Markovian-Embedding/data/processed/NS_data.csv"
UT_NS_PATH = "/Users/mathisvernier/Markovian-Embedding/data/processed/UT_NS_data.csv"


if __name__ == "__main__":
    loader = DataLoader(RAW_PATH)

    # we write the data 3 times because we need different versions (untransformed, not normalised, everything)

    # 1. Transformed + standardized (for PCA / Koopman)
    data, tcodes = loader.load_data()
    processor = DataProcessor(data)
    X_std, mu, sd = processor.process(data, tcodes, standardize=True)
    loader.save(X_std, PROCESSED_PATH)

    # 2. Transformed, NOT standardized (for univariate/bivariate stats, real units)
    X_ns = processor.process(data, tcodes, standardize=False)
    loader.save(X_ns, NS_PATH)

    # 3. Untransformed raw levels, cleaned + outlier-adjusted (for cointegration)
    X_ut = processor.clean_data(data)          # no transform() call, keep raw levels (non stationary)
    X_ut = processor.adjust_outliers(X_ut)
    loader.save(X_ut, UT_NS_PATH)

    ##plots for qualitative data description

    df, tc = loader.load_data()
    level = df["UNRATE"]
    diff = level.diff()

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
