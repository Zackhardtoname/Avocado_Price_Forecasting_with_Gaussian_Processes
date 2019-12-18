# train = df[(df.index.year != 2016) & (df.index.year != 2018)].AveragePrice.values.reshape(-1, 1)
# ground_truth = df[df.index.year == 2016].AveragePrice.values.reshape(-1, 1)
# + Kernels.RBF() \
    # + Kernels.Matern() \
k = y_gpr + test.mean() - y_gpr.mean()
sklearn.metrics.r2_score(k, test)

more_X_train = np.arange(925, 2000, 7).reshape(-1, 1)
more_y_val, more_y_std = gpr.predict(more_X_train, return_std=True)

plt.plot(more_y_val)
plt.show()

# filter outliers for California and TotalUS
if region == "California" or "TotalUS":
    df = df[df["AveragePrice"] != 1]

if test_year:
    df = os.path.join('.', 'data', type, 'raw', f'{region}.csv')
    y_train = df[df.index.year < test_year].AveragePrice.values.reshape(-1, 1)
    y_test = df[df.index.year == test_year].AveragePrice.values.reshape(-1, 1)

# ax.set_ylim(bottom=np.min(np.concatenate((y_train, y_test))), top=np.max(np.concatenate((y_train, y_test))))
