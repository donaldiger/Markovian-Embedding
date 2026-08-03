import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

## descriptive evidence

#groups mapping

PROCESSED_PATH = "/Users/mathisvernier/Markovian-Embedding/data/processed/processed_FRED_MD.csv"
APPENDIX_PATH = "/Users/mathisvernier/Markovian-Embedding/data/raw/FRED-MD_historic_appendix.csv"

GROUP_NAMES = {
    1: "Output and Income",
    2: "Labor Market",
    3: "Housing",
    4: "Consumption, Orders & Inventories",
    5: "Money and Credit",
    6: "Interest and Exchange Rates",
    7: "Prices",
    8: "Stock Market",
}

NAME_FIXES = {
    "IPB51222s": "IPB51222S",
}

def build_mapping(appendix_path=APPENDIX_PATH):
    appx = pd.read_csv(appendix_path, encoding="latin1")  
    appx["fred"] = appx["fred"].replace(NAME_FIXES)
    appx["group_name"] = appx["group"].map(GROUP_NAMES)
 
    mapping = (
        appx[["fred", "description", "group", "group_name", "tcode"]]
        .rename(columns={"fred": "variable"})
    )
    return mapping

#print(build_mapping()) #has more variables since the appendix also has variables that arent used anymore by FREDMD (+non cleaned)


def main_file(processed_path=PROCESSED_PATH, mapping=None):

    if mapping is None: 
        mapping = build_mapping()
    data = pd.read_csv(processed_path)
    data["sasdate"] = pd.to_datetime(data["sasdate"])
    return data

print(main_file())




