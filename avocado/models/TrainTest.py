import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics
import sklearn.gaussian_process.kernels as Kernels
from sklearn.gaussian_process import GaussianProcessRegressor

import avocado.utils as utils
 
region = 'WestTexNewMexico'
df, X_train, y_train, X_test, y_test = utils.get_data(region=region)


# periodicity of 358 is 1 year. performs poorly on test
# Best: 100., 200.
# specify the kernel functions; please see the paper for the rationale behind the choices
gp_kernel = Kernels.ExpSineSquared(20., periodicity=358., periodicity_bounds=(1e-2, 1e8)) \
    + 0.8 * Kernels.RationalQuadratic(alpha=20., length_scale=80.) \
    + Kernels.WhiteKernel(1e2)

# + Kernels.ExpSineSquared(20., periodicity=158., periodicity_bounds=(1e-2, 1e8)) \
# + Kernels.ExpSineSquared(20., periodicity=79., periodicity_bounds=(1e-2, 1e8)) \
# + Kernels.ExpSineSquared(20., periodicity=30., periodicity_bounds=(1e-2, 1e8)) \

gpr = GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True, n_restarts_optimizer=10)
gpr.fit(X_train, y_train)

# Predict using gaussian process regressor
y_gpr_train, y_std_train = gpr.predict(X_train, return_std=True)
y_gpr_test, y_std_test = gpr.predict(X_test, return_std=True)

# Print the results
y_gpr_train = y_gpr_train.flatten()
y_gpr_test = y_gpr_test.flatten()
y_train = y_train.flatten()
y_test = y_test.flatten()
X_train = X_train.flatten()
X_test = X_test.flatten()

rmse_train = np.sqrt(((y_gpr_train - y_train) ** 2).mean())
r_2_train = sklearn.metrics.r2_score(y_gpr_train, y_train)

rmse_test = np.sqrt(((y_gpr_test - y_test) ** 2).mean())
r_2_test = sklearn.metrics.r2_score(y_gpr_test, y_test)

print(f'Train RMSE: {rmse_train:.3f}')
print(f'Train R2 score: {r_2_train:.3f}')
print(f'Test RMSE: {rmse_test:.3f}')
print(f'Test R2 score: {r_2_test:.3f}')
print(f'Predicted mean: {y_gpr_test.mean():.3f}\n')
print(f'Test mean: {y_test.mean():.3f}\n')

# Plotting the 95% posterior predictive checking interval
z = 1.96
CI_lower_bound_train = y_gpr_train - z * y_std_train
CI_higher_bound_train = y_gpr_train + z * y_std_train
out_of_CI_ptc_train = np.mean((y_train < CI_lower_bound_train) | (y_train > CI_higher_bound_train)) * 100

CI_lower_bound_test = y_gpr_test - z * y_std_test
CI_higher_bound_test = y_gpr_test + z * y_std_test
out_of_CI_ptc_test = np.mean((y_test < CI_lower_bound_test) | (y_test > CI_higher_bound_test)) * 100

print(f'Train CI ptc: {out_of_CI_ptc_train:.3f}')
print(f'Test CI ptc: {out_of_CI_ptc_test:.3f}')
print(f'Model: {str(gpr.kernel_)}')

# Plot the train and test results
utils.plot(region,
           df,
           X_train, y_train, 
           X_test, y_test,
           y_gpr_train, y_gpr_test,
           y_std_train, y_std_test,
           CI_lower_bound_train, CI_higher_bound_train,
           CI_lower_bound_test, CI_higher_bound_test)

