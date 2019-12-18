import numpy as np
import sklearn.metrics
from sklearn.gaussian_process import GaussianProcessRegressor

import avocado.utils as utils


def run_gp(kernel, 
           n_restarts_optimizer=10, 
           type_='organic',
           region='WestTexNewMexico'):
    """
    Parameters
    ----------
    kernel : sklearn.gaussian_process.kernels
        Defined kernel function used to define a GP model

    n_restarts_optimizer : int
        Number of times to restart the kernel hyperparameter search

    type_ : str
        conventional or organic

    region : str
        US region e.g. SanFrancisco

    Effects
    -------
    Train a GP model with the specified kernel function
    on the data type_/region. Prints model statistics and
    generates publication quality image of model results.

    """
    

    # Load partitioned data set
    df, X_train, y_train, X_test, y_test = utils.get_data(type_=type_, region=region)

    # Define GP model with kernel function. 
    # Set prior mean to train sample mean.
    # Try n_restarts_optimizer different random samples from the hyperparameter space
    # Random state for reproducibility
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=n_restarts_optimizer, random_state=451)

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
    utils.plot(type_,
               region,
               df,
               X_train, y_train, 
               X_test, y_test,
               y_gpr_train, y_gpr_test,
               y_std_train, y_std_test,
               CI_lower_bound_train, CI_higher_bound_train,
               CI_lower_bound_test, CI_higher_bound_test)
