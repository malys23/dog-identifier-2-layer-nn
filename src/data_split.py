#A new data splitting function
#Splits images and labels so train and test sets keep the same share of each class
import numpy as np

def stratified_split(x, y, test_fraction=0.2, seed=1):
    rng = np.random.default_rng(seed)
    y = np.asarray(y).ravel()
    train_idx, test_idx = [], []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        n_test = round(len(idx) * test_fraction)
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]