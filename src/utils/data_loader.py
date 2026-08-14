import numpy  as np
import pandas as pd

#data_path = /Users/mathisvernier/Markovian-Embedding/data/raw/2025-09-fred-md.csv

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = pd.DataFrame()
        self.tcodes = pd.DataFrame()
        self.TCODE_OVERRIDES = { #only for permit rn might have more later
                                "PERMIT": 5,
                                }

    def load_data(self):
        """Load raw FRED-MD csv: first data row holds the tcodes, rest is the series."""
        self.data = pd.read_csv(self.data_path, sep=";")
        self.tcodes = self.data.iloc[0, 1:].astype(float).astype(int) #extract the codes of the data
        self.data = self.data.iloc[1:].copy()
        self.data["sasdate"] = pd.to_datetime(self.data["sasdate"], format="%m/%d/%Y")
        self.data = self.data.set_index("sasdate").apply(pd.to_numeric, errors="coerce") #data itself

        for series, code in self.TCODE_OVERRIDES.items():
            if series in self.tcodes.index:
                self.tcodes[series] = code

        return self.data, self.tcodes

    def save(self, X, path):
        X.to_csv(path)
