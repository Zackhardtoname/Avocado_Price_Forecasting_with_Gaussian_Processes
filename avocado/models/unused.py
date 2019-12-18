# train = df[(df.index.year != 2016) & (df.index.year != 2018)].AveragePrice.values.reshape(-1, 1)
# ground_truth = df[df.index.year == 2016].AveragePrice.values.reshape(-1, 1)

k = y_gpr + test.mean() - y_gpr.mean()
sklearn.metrics.r2_score(k, test)