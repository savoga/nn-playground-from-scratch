'''
- Data generation: moons, circles, blobs
- Loop with predictions
'''

import pandas as pd

from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons

def generateData(data_pattern='blobs'):
    df_data = pd.DataFrame()
    if data_pattern=='blobs':
        X_train, y_train = make_blobs(n_samples=100,
                              n_features=2,
                              centers=2,
                              cluster_std=0.2,
                              center_box=(0,5))
    elif data_pattern=='circles':
        X_train, y_train = make_circles(n_samples=100,
                              noise=0.2,
                              factor=0.2)
    elif data_pattern=='moons':
        X_train, y_train = make_moons(n_samples=100,
                              noise=.05)
    df_data['x'] = X_train[:,0]
    df_data['y'] = X_train[:,1]
    df_data['cluster'] = y_train
    return df_data