import numpy as np
import sklearn.metrics
import sklearn.gaussian_process.kernels as Kernels
from sklearn.gaussian_process import GaussianProcessRegressor

import avocado.utils as utils

type_ = 'organic' 
region = 'WestTexNewMexico'


# periodicity of 358 is 1 year. performs poorly on test
# Best: 100., 200.
# specify the kernel functions; please see the paper for the rationale behind the choices
gp_kernel = Kernels.ExpSineSquared(20., periodicity=358., periodicity_bounds=(1e-2, 1e8)) \
    + 0.8 * Kernels.RationalQuadratic(alpha=20., length_scale=80.) \
    + Kernels.WhiteKernel(1e2)

df, X_train, y_train, X_test, y_test = utils.get_data(type_=type_, region=region)


# + Kernels.ExpSineSquared(20., periodicity=158., periodicity_bounds=(1e-2, 1e8)) \
# + Kernels.ExpSineSquared(20., periodicity=79., periodicity_bounds=(1e-2, 1e8)) \
# + Kernels.ExpSineSquared(20., periodicity=30., periodicity_bounds=(1e-2, 1e8)) \

# Define GP model with kernel function. 
# Set prior mean to train sample mean.
# Try 10 different random samples from the hyperparameter space
gpr = GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True, n_restarts_optimizer=10)

# Fit GP model on training data
gpr.fit(X_train, y_train)

# Predict using gaussian process regressor
y_gpr_train, y_std_train = gpr.predict(X_train, return_std=True)
y_gpr_test, y_std_test = gpr.predict(X_test, return_std=True)

# Flatten arrays for math and plotting
y_gpr_train = y_gpr_train.flatten()
y_gpr_test = y_gpr_test.flatten()
y_train = y_train.flatten()
y_test = y_test.flatten()
X_train = X_train.flatten()
X_test = X_test.flatten()

# Calculate rmse, r_2 for train set
rmse_train = np.sqrt(((y_gpr_train - y_train) ** 2).mean())
r_2_train = sklearn.metrics.r2_score(y_gpr_train, y_train)

# Calculate rmse, r_2 for test set
rmse_test = np.sqrt(((y_gpr_test - y_test) ** 2).mean())
r_2_test = sklearn.metrics.r2_score(y_gpr_test, y_test)

# Print model results
print(f'Train RMSE: {rmse_train:.3f}')
print(f'Train R2 score: {r_2_train:.3f}')
print(f'Test RMSE: {rmse_test:.3f}')
print(f'Test R2 score: {r_2_test:.3f}')
print(f'Predicted mean: {y_gpr_test.mean():.3f}\n')
print(f'Test mean: {y_test.mean():.3f}\n')

# Generate the 95% posterior predictive check interval
z = 1.96
CI_lower_bound_train = y_gpr_train - z * y_std_train
CI_higher_bound_train = y_gpr_train + z * y_std_train
out_of_CI_ptc_train = np.mean((y_train < CI_lower_bound_train) | (y_train > CI_higher_bound_train)) * 100

CI_lower_bound_test = y_gpr_test - z * y_std_test
CI_higher_bound_test = y_gpr_test + z * y_std_test
out_of_CI_ptc_test = np.mean((y_test < CI_lower_bound_test) | (y_test > CI_higher_bound_test)) * 100

print(f'Train out of CI ptc: {out_of_CI_ptc_train:.3f}')
print(f'Test out of CI ptc: {out_of_CI_ptc_test:.3f}')

# Print model with kernel functions and associated hyperparameters
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
