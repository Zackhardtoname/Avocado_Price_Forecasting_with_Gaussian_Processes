import os
import numpy as np
import pandas as pd


def get_data(type_='organic', region='TotalUS'):
    """
    Parameters
    ----------
    type_ : str
        conventional or organic

    region : str
        US region e.g. SanFrancisco

    Returns
    -------
    X_train, y_train : np.ndarrry
        Arrays containing train data for 
        `type` and `region` specified in input

    """

    # Define paths to data
    train_path = os.path.join('.', 'data', type_, 'train', f'{region}.csv')
    test_path = os.path.join('.', 'data', type_, 'test', f'{region}.csv')

    # Read data into pandas dataframe
    df_train = pd.read_csv(train_path, index_col='Date')
    df_test = pd.read_csv(test_path, index_col='Date')

    # Collect underlying np arrays from dataframes
    y_train = df_train.AveragePrice.values.reshape(-1, 1)
    y_test = df_test.AveragePrice.values.reshape(-1, 1)

    # Generate weekly-spaced data for each price observation (incremented by 7 starting from 1)
    X_train = np.array([7 * i + 1 for i in range(len(y_train))]).reshape(-1, 1)
    X_test = np.array([7 * i + 1 for i in range(len(y_train), len(y_train) + len(y_test))]).reshape(-1, 1)

    # Join train/test dataframes into one for plotting purposes
    df = pd.concat((df_train, df_test), join="inner")
    df.index = pd.to_datetime(df.index)

    return df, X_train, y_train, X_test, y_test
