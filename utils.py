from typing import Tuple
import numpy as np
import json

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from utils import *

def load_data(path=None, test_split=0.2, target_as_onehot=False, 
                scaling_save_path=None, scaling_load_path=None) -> Tuple[np.ndarray, np.ndarray]:
    if path is None:
        path = './data.csv'
    df = pd.read_csv(path, header=None)
    df.pop(2)
    Y_raw = (df.pop(1) == 'M').astype(int).to_numpy()

    if scaling_load_path is not None:
        conf = np.load(scaling_load_path, allow_pickle=True).item()
        X, max_, min_ = minmax_scale(df.to_numpy(), max_=conf['max'], min_=conf['min'])
    else:
        X, max_, min_ = minmax_scale(df.to_numpy())
    if scaling_save_path is not None:
        np.save(scaling_save_path, {'max': max_, 'min': min_}, allow_pickle=True)

    if target_as_onehot:
        Y = np.zeros((Y_raw.shape[0], 2))
        np.put_along_axis(Y, Y_raw.reshape((-1, 1)), 1, 1)
    else:
        Y = Y_raw

    if test_split > 0:
        sss = StratifiedShuffleSplit(1, test_size=test_split, random_state=21)
        train_idx, test_idx = next(sss.split(X, Y_raw))
        result = X[train_idx], Y[train_idx], X[test_idx], Y[test_idx]
    else:
        result = [X, Y]
    # print('Data stats:', [k.shape for k in result], Y.shape, X.max(), X.min(), np.max(max_), np.min(min_))
    return result

def minmax_scale(vals: np.ndarray, max_=None, min_=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_ = np.min(vals, axis=0) if min_ is None else min_
    max_ = np.max(vals, axis=0) if max_ is None else max_

    return (vals - min_) / (max_ - min_), max_, min_