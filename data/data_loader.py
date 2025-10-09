import os
import numpy as np
import pandas as pd
from typing import Optional
from ucimlrepo import fetch_ucirepo

_cached_data: dict[str, Optional[np.ndarray]] = {'X': None, 'y': None}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "diabetes_noise_30features.csv")
ORIGINAL_FILE = os.path.join(BASE_DIR, "kaggle_diabetes_dataset_original.csv")

LABEL = "Diabetes_binary"

def create_noisy_dataset():
    df = pd.read_csv(ORIGINAL_FILE)

    for i in range(9):
        df[f'noise_{i+1}'] = np.round(np.random.uniform(0, 1, len(df)), 6)

    df.to_csv(DATA_FILE, index=False)
    print(f"Created dataset with {len(df.columns)-1} features.")


def load_data():
    global _cached_data

    if _cached_data['X'] is not None:
        return _cached_data['X'], _cached_data['y']

    if not os.path.exists(DATA_FILE):
        create_noisy_dataset()

    df = pd.read_csv(DATA_FILE)
    _cached_data['X'] = df.drop(LABEL, axis=1).values
    _cached_data['y'] = df[LABEL].values

    return _cached_data['X'], _cached_data['y']


def create_noisy_iris():
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target

    for i in range(5):
        df[f'noise_{i + 1}'] = np.round(np.random.uniform(0, 1, len(df)), 6)

    return df


def load_noisy_iris():
    global _cached_data

    if _cached_data['X'] is not None:
        return _cached_data['X'], _cached_data['y']

    df = create_noisy_iris()
    _cached_data['X'] = df.drop('target', axis=1).values
    _cached_data['y'] = df['target'].values

    return _cached_data['X'], _cached_data['y']

def create_clean_iris():
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target


    return df


def load_clean_iris():
    global _cached_data

    if _cached_data['X'] is not None:
        return _cached_data['X'], _cached_data['y']

    df = create_clean_iris()
    _cached_data['X'] = df.drop('target', axis=1).values
    _cached_data['y'] = df['target'].values

    return _cached_data['X'], _cached_data['y']

def load_breast_cancer():
    global _cached_data

    if _cached_data['X'] is not None:
        return _cached_data['X'], _cached_data['y']

    dataset = fetch_ucirepo(id=17)
    _cached_data['X'] = dataset.data.features.values
    _cached_data['y'] = dataset.data.targets.values.ravel()

    return _cached_data['X'], _cached_data['y']
