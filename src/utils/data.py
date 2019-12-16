import os
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
    train, test : np.ndarrry
        Arrays containing train and test data for 
        `type` and `region` specified in input

    """

    train_path = os.path.join('.','data', type_, 'train', region)
    train_path = os.path.join('.','data', type_, 'test', region)

    df_train = pd.read_csv(train_path, index_col='Date')
    df_test = pd.read_csv(test_path, index_col='Date')

    return df_train.AveragePrice.values, df_test.AveragePrice.values
