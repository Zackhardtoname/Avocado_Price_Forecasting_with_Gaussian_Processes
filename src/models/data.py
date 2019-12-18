import os
import numpy as np
import pandas as pd

def get_data(type='organic', region='TotalUS'):
    train_path = os.path.join('.', 'data', type, 'train', f'{region}.csv')
    test_path = os.path.join('.', 'data', type, 'test', f'{region}.csv')

    df_train = pd.read_csv(train_path, index_col='Date')
    df_train.index = pd.to_datetime(df_train.index)

    df_test = pd.read_csv(test_path, index_col='Date')
    df_test.index = pd.to_datetime(df_test.index)

    y_train = df_train.AveragePrice.values.reshape(-1, 1)
    y_test = df_test.AveragePrice.values.reshape(-1, 1)

    X_train = np.array([7 * i + 1 for i in range(len(y_train))]).reshape(-1, 1)
    X_test = np.array([7 * i + 1 for i in range(len(y_train), len(y_train) + len(y_test))]).reshape(-1, 1)

    df = pd.concat((df_train, df_test), join="inner")
    df.index = pd.to_datetime(df.index)

    return df, X_train, y_train, X_test, y_test