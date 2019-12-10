
import numpy as np

import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as Kernels
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
    #test_path = os.path.join('.','data', type_, 'ground_truth', f'{region}.csv')

    df_train = pd.read_csv(train_path, index_col='Date')

    return df_train

df = get_data(region='California')
df = df[df["AveragePrice"] != 1]
df.index = pd.to_datetime(df.index)
df.sort_index(inplace=True)
print(df.groupby(df.index.year).describe())

# 2018 only has 12 data points
last_year = df.index.year.unique()[-2]
train = df[df.index.year < last_year].AveragePrice.values.reshape(-1, 1)
ground_truth = df[df.index.year == last_year].AveragePrice.values

base_range = list(range(1, 365, 7))
X = []

for _ in range(len(train) // 52):
    X += base_range
train = train[:(len(train) // 52) * 52, ]
y = train
X = np.array(X).reshape(-1, 1)
# assert len(X) == len(y)

gp_kernel = Kernels.ExpSineSquared() \
    + Kernels.WhiteKernel() \
    + Kernels.RBF() \
    + Kernels.Matern() \
    + Kernels.RationalQuadratic() \

gp_kernel = Kernels.ExpSineSquared(2.0, 6.0, periodicity_bounds=(1e-2, 1e5)) \
    + Kernels.WhiteKernel(1e-1)

gpr = GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True)
gpr.fit(X, y)

# Predict using gaussian process regressor
y_gpr, y_std = gpr.predict(np.array(base_range).reshape(-1, 1), return_std=True)
# y_gpr = np.array(y_gpr)
# y_std = np.array(y_std)
# Plot results
y_gpr = y_gpr.flatten()

plt.figure(figsize=(10, 5))
plt.scatter(X, y, c='k', label='old data')
plt.scatter(base_range, ground_truth[:len(base_range)], c='r', label='new data')
plt.plot(base_range, y_gpr, color='darkorange', lw=2,
         label='GPR (%s)' % gpr.kernel_)
z = 1.96
CI_lower_bound = y_gpr - z * y_std
CI_higher_bound = y_gpr + z * y_std

plt.fill_between(base_range, CI_lower_bound, CI_higher_bound, color='blue', alpha=0.2)
plt.xlabel('data')
plt.ylabel('target')
plt.title('GPR')
plt.legend(loc="best",  scatterpoints=1, prop={'size': 8})
plt.show()

test_size = len(y_gpr)
ground_truth = ground_truth[:test_size]

rmse = np.linalg.norm(y_gpr - ground_truth) / np.sqrt(test_size)
r_2 = sklearn.metrics.r2_score(y_gpr, ground_truth)
r = scipy.stats.pearsonr(y_gpr, ground_truth)

out_of_CI_ctn = 0

for i in range(len(ground_truth)):
    if ground_truth[i] < CI_lower_bound[i] or ground_truth[i] > CI_higher_bound[i]:
        out_of_CI_ctn += 1

out_of_CI_ptc = out_of_CI_ctn / len(ground_truth) * 100