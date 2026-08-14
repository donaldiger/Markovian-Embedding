from src.utils import DataLoader, DataProcessor

#from src.var     import VAR

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import decomposition
from sklearn import datasets


RAW_PATH = "/Users/mathisvernier/Markovian-Embedding/data/raw/2025-09-fred-md.csv"

SELECTED = ["RPI", "DPCERA3M086SBEA", "UNRATE", "UEMPMEAN", "PERMIT",
            "ISRATIOx", "M1SL", "REALLN", "S&P PE ratio", "GS10",
            "EXSZUSx", "CUSR0000SAC", "CES2000000008"]



if __name__ == "__main__":

    loader = DataLoader(RAW_PATH)
    data, tcodes = loader.load_data()

    processor = DataProcessor(data)
    X_std, mu, sd = processor.process(data, tcodes)

    data = X_std[SELECTED].copy()
    u, sigma, w_transpose = np.linalg.svd(data, full_matrices=False)
    
    """
    #we first visualize it usinf 2 principal components
    W2 = w_transpose[:2, :].T 
    T_2 = data @ W2
    T_2.shape

    plt.figure(figsize=(10, 7))
    plt.scatter(T_2[:2000, 0], T_2[:2000, 1], alpha=0.7)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.colorbar(label='Variable')
    plt.title('Data Projected onto 2 Principal Components')
    plt.show()
    # Load data
    # Apply PCA
    # Train VAR model
    # Reconstruct predictions in observation space
    """
    
    pass