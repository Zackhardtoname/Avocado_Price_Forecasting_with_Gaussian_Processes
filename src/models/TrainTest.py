
import numpy as np

import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared
import sklearn.metrics
import scipy.stats
import os
import pandas as pd


import datetime
 
# MM/DD/YY HH:MM:SS 
#datetime_obj = datetime.datetime.strptime(datetime_str, '%y/%m/%d')

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

    train_path = os.path.join('../../','data', type_, 'raw', f'{region}.csv')
    #test_path = os.path.join('.','data', type_, 'test', f'{region}.csv')

    df_train = pd.read_csv(train_path, index_col='Date')

    return df_train

df = get_data(region='TotalUS')
df = df[df["AveragePrice"] != 1]
df.index = pd.to_datetime(df.index)
df.sort_index(inplace=True)

# 2018 only has 12 data points
last_year = df.index.year.unique()[-2]
train = df[df.index.year < last_year].AveragePrice.values.reshape(-1, 1)
test = df[df.index.year == last_year].AveragePrice.values.reshape(-1, 1)

base_range = list(range(1, 365, 7))
X = []

for _ in range(len(train) // 52):
    X += base_range
train = train[:(len(train) // 52) * 52, ]
y = train
X = np.array(X).reshape(-1, 1)
# assert len(X) == len(y)

gp_kernel = ExpSineSquared(2.0, 6.0, periodicity_bounds=(1e-2, 1e5)) \
    + WhiteKernel(1e-1)
gpr = GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True)
gpr.fit(X, y)

# Predict using gaussian process regressor
y_gpr, y_std = gpr.predict(np.array(base_range).reshape(-1, 1), return_std=True)
# y_gpr = np.array(y_gpr)
# y_std = np.array(y_std)
# Plot results
y_gpr = y_gpr.tolist()
y_std = list(y_std)
plt.figure(figsize=(10, 5))
plt.scatter(X, y, c='k', label='old data')
plt.scatter(base_range, test[:len(base_range)], c='r', label='new data')
plt.plot(base_range, y_gpr, color='darkorange', lw=2,
         label='GPR (%s)' % gpr.kernel_)
plt.fill_between(base_range, [a_i[0] - b_i for a_i, b_i in zip(y_gpr, y_std)],
                 [a_i[0] + b_i for a_i, b_i in zip(y_gpr, y_std)], color='blue',
                 alpha=0.2)
plt.xlabel('data')
plt.ylabel('target')
# plt.xlim(X.min(), X.max())
# plt.ylim(y.min(), y.max())
plt.title('GPR')
plt.legend(loc="best",  scatterpoints=1, prop={'size': 8})
plt.show()

rmse = np.linalg.norm(y_gpr - test[:len(y_gpr)]) / np.sqrt(len(y_gpr))
r_2 = sklearn.metrics.r2_score(y_gpr, test[:len(y_gpr)])
r = scipy.stats.pearsonr(y_gpr, test[:len(y_gpr)])