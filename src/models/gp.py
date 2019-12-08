
import numpy as np

import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared

import os
import pandas as pd


import datetime
 
# MM/DD/YY HH:MM:SS 
#datetime_obj = datetime.datetime.strptime(datetime_str, '%y/%m/%d')

scale = 21

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

    train_path = os.path.join('.','data', type_, 'train', f'{region}.csv')
    #test_path = os.path.join('.','data', type_, 'test', f'{region}.csv')

    df_train = pd.read_csv(train_path, index_col='Date')

    return np.linspace(0, scale, len(df_train)).reshape(-1, 1), df_train.AveragePrice.values


X, y = get_data(region='TotalUS')


print(X.shape)
print(y.shape)


gp_kernel = ExpSineSquared(2.0, 6.0, periodicity_bounds=(1e-2, 1e5)) \
    + WhiteKernel(1e-1)
gpr = GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True)
gpr.fit(X, y)


# Predict using gaussian process regressor
y_gpr, y_std = gpr.predict(X, return_std=True)


# Plot results
plt.figure(figsize=(10, 5))
plt.scatter(X, y, c='k', label='data')
plt.plot(X, y_gpr, color='darkorange', lw=2,
         label='GPR (%s)' % gpr.kernel_)
plt.fill_between(X[:, 0], y_gpr - y_std, y_gpr + y_std, color='darkorange',
                 alpha=0.2)
plt.xlabel('data')
plt.ylabel('target')
plt.xlim(X.min(), X.max())
plt.ylim(y.min(), y.max())
plt.title('GPR')
plt.legend(loc="best",  scatterpoints=1, prop={'size': 8})
plt.show()
