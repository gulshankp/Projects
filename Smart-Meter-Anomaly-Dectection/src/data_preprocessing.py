import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def load_data(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df

def preprocess(df, window_size=24):
    scaler = MinMaxScaler()
    df["scaled"] = scaler.fit_transform(df[["consumption"]])
    
    X, y = [], []
    
    for i in range(len(df) - window_size):
        X.append(df["scaled"].values[i:i+window_size])
        y.append(df["scaled"].values[i+window_size])
    

    X = np.array(X)
    y = np.array(y)
    
    return X, y, scaler, df