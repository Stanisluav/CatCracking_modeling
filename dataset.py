import pandas as pd
import torch
import numpy as np

def load_experimental_data(file_path="experimental_data.txt"):
    df = pd.read_csv(file_path, header=0)
    # Вычисляем VGO = 100 - X_VGO
    df['VGO'] = 100.0 - df['X_VGO']
    df = df[(df['VGO'] >= 0) & (df['Light_Gases'] >= 0) & (df['Gasoline'] >= 0) &
            (df['Light_Cycle_Oil'] >= 0) & (df['Coke'] >= 0)]
    experiments = []
    for idx, row in df.iterrows():
        experiments.append({
            'C_O': row['C_O'],
            't_s': row['t_s'],
            'T_C': row['T_C'],
            'target': np.array([row['VGO'], row['Light_Cycle_Oil'], 
                                row['Gasoline'], row['Light_Gases'], row['Coke']], dtype=np.float32)
        })
    return experiments

def normalize_concentrations(c):
    return c / 100.0

def denormalize_concentrations(c_norm):
    return c_norm * 100.0