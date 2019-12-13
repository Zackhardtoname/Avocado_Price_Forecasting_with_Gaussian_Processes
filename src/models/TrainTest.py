
import numpy as np

import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as Kernels
import sklearn.metrics
import scipy.stats
import os
import pandas as pd


import datetime
 

def get_data(type_='organic', region='TotalUS', test_year=2017):
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

    train_path = os.path.join('.', 'data', type_, 'raw', f'{region}.csv')

    df = pd.read_csv(train_path, index_col='Date')
    df.index = pd.to_datetime(df.index)

    # filter outliers for California
    #if region == "California" or "TotalUS":
    #    df = df[df["AveragePrice"] != 1]

    y_train = df[df.index.year < test_year].AveragePrice.values.reshape(-1, 1)
    y_test = df[df.index.year == test_year].AveragePrice.values.reshape(-1, 1)

    X_train = np.array([7 * i + 1 for i in range(len(y_train))]).reshape(-1, 1)
    X_test = np.array([7 * i + 1 for i in range(len(y_train), len(y_train) + len(y_test))]).reshape(-1, 1)

    return df, X_train, y_train, X_test, y_test


region = 'California'
df, X_train, y_train, X_test, y_test = get_data(region=region)

print(df)


#print(df.head())
#print(df.groupby(df.index.year).describe())


# gp_kernel = Kernels.ExpSineSquared(2.0, 6.0, periodicity_bounds=(1e-2, 1e5)) \
#     + Kernels.WhiteKernel() \
#     + Kernels.RBF() \
#     + Kernels.Matern() \
#     + Kernels.RationalQuadratic() \
#     + Kernels.WhiteKernel(1e-1)

gp_kernel = Kernels.ExpSineSquared(100., 200., periodicity_bounds=(1e-2, 1e8)) \
    + Kernels.WhiteKernel(1e1)

gpr = GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True)
gpr.fit(X_train, y_train)

# Predict using gaussian process regressor
y_gpr_train, y_std_train = gpr.predict(X_train, return_std=True)
y_gpr_test, y_std_test = gpr.predict(X_test, return_std=True)

# Plot results
y_gpr_train = y_gpr_train.flatten()
y_gpr_test = y_gpr_test.flatten()
y_train = y_train.flatten()
y_test = y_test.flatten()
X_train = X_train.flatten()
X_test = X_test.flatten()


rmse_train = np.sqrt(((y_gpr_train - y_train) ** 2).mean())
r_2_train = sklearn.metrics.r2_score(y_gpr_train, y_train)
r_train = scipy.stats.pearsonr(y_gpr_train, y_train)

rmse_test = np.sqrt(((y_gpr_test - y_test) ** 2).mean())
r_2_test = sklearn.metrics.r2_score(y_gpr_test, y_test)
r_test = scipy.stats.pearsonr(y_gpr_test, y_test)

print(f'Train RMSE: {rmse_train}')
print(f'Train R2 score: {r_2_train}')
print(f'Train R score: {r_train}\n')
print(f'Test RMSE: {rmse_test}')
print(f'Test R2 score: {r_2_test}')
print(f'Test R score: {r_test}\n')

z = 1.96
CI_lower_bound_train = y_gpr_train - z * y_std_train
CI_higher_bound_train = y_gpr_train + z * y_std_train

out_of_CI_ptc_train = np.sum((y_train < CI_lower_bound_train) | (y_train > CI_higher_bound_train)) / len(y_train) * 100

CI_lower_bound_test = y_gpr_test - z * y_std_test
CI_higher_bound_test = y_gpr_test + z * y_std_test

out_of_CI_ptc_test = np.sum((y_test < CI_lower_bound_test) | (y_test > CI_higher_bound_test)) / len(y_test) * 100

print(f'Train CI ptc: {out_of_CI_ptc_train}')
print(f'Test CI ptc: {out_of_CI_ptc_test}')

# Plotting

fig, ax = plt.subplots(figsize=(12, 7))


plt.scatter(X_train, y_train, c='k', label='2015-2016 Price')
plt.scatter(X_test, y_test, c='r', label='2017 Price (out-of-sample)')
plt.plot(X_train, y_gpr_train, color='darkorange', lw=2, label='GP: ' + "\n       +".join(str(gpr.kernel_).split("+")))
plt.plot(X_test, y_gpr_test, color='darkorange', lw=2)
plt.fill_between(X_train, CI_lower_bound_train, CI_higher_bound_train, color='blue', alpha=0.2)
plt.fill_between(X_test, CI_lower_bound_test, CI_higher_bound_test, color='blue', alpha=0.2)
plt.xlabel('Time (Weekly)')
plt.ylabel('Average Avacado Price (USD)')
plt.title(f'GPR: {region} Organic Avacado Price')

xticks = list(df.index[df.index.year < 2018].strftime('%m-%d-%Y'))
# Only show every 10th tick
for i in range(len(xticks)):
    if i % 10:
        xticks[i] = ''

plt.xticks(np.concatenate((X_train, X_test)), labels=xticks)

# Rotates x axis date labels by 45 degrees
for label in ax.xaxis.get_ticklabels():
    label.set_rotation(45)

#ax.set_ylim(bottom=np.min(np.concatenate((y_train, y_test))), top=np.max(np.concatenate((y_train, y_test))))

plt.legend(loc='upper left', scatterpoints=1, prop={'size': 11})


#plt.grid(which='both', alpha=0.5)
plt.grid(linewidth=0.25, alpha=0.5)
plt.subplots_adjust(bottom=0.22)
plt.text(5, 5, f"{100 - out_of_CI_ptc_train:.2f}% of out-of-sample data points are inside PPCI")
plt.savefig('figures/GPR_.png')
plt.show()
