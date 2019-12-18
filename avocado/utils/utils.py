import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

    # Define paths to data
    train_path = os.path.join('.', 'data', type_, 'train', f'{region}.csv')
    test_path = os.path.join('.', 'data', type_, 'test', f'{region}.csv')

    # Read data into pandas dataframe
    df_train = pd.read_csv(train_path, index_col='Date')
    df_test = pd.read_csv(test_path, index_col='Date')

    # Collect underlying np arrays from dataframes
    y_train = df_train.AveragePrice.values.reshape(-1, 1)
    y_test = df_test.AveragePrice.values.reshape(-1, 1)

    # Generate weekly-spaced data for each price observation (incremented by 7 starting from 1)
    X_train = np.array([7 * i + 1 for i in range(len(y_train))]).reshape(-1, 1)
    X_test = np.array([7 * i + 1 for i in range(len(y_train), len(y_train) + len(y_test))]).reshape(-1, 1)

    # Join train/test dataframes into one for plotting purposes
    df = pd.concat((df_train, df_test), join="inner")
    df.index = pd.to_datetime(df.index)

    return df, X_train, y_train, X_test, y_test


def plot(type_,
         region,
         df,
         X_train, y_train, 
         X_test, y_test,
         y_gpr_train, y_gpr_test,
         y_std_train, y_std_test,
         CI_lower_bound_train, CI_higher_bound_train,
         CI_lower_bound_test, CI_higher_bound_test):

    """
    Parameters
    ----------
    type_ : str
        conventional or organic

    region : str
        US region e.g. SanFrancisco

    df : pd.DataFrame
        contains datetime info

    X_train, y_train, X_test, y_test : np.ndarray
        contains train/test data

    y_gpr_train, y_gpr_test : np.ndarray
        contains train/test GP model predictions

    y_std_train, y_std_test : np.ndarray
        contains train/test model uncertainty

    CI_* : np.ndarray
        contain 95% credible intervals for train/test predictions

    Effects
    -------
    Plots publication quality image displaying train/test data
    along side model predictions and uncertainty bands.

    """

    # Plot the ground truth and predictions
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot training data as black points
    plt.scatter(X_train, y_train, c='k', s=10, label='Train')
    # Plot testing data as red points
    plt.scatter(X_test, y_test, c='r', s=10, label='Test')
    # Plot GP train prediction as an orange line
    plt.plot(X_train, y_gpr_train, color='darkorange', lw=2, label='GP')
    # Plot GP test prediction as an orange line
    plt.plot(X_test, y_gpr_test, color='darkorange', lw=2)
    # Plot blue bands for GP train 95% PPC interval
    plt.fill_between(X_train, CI_lower_bound_train, CI_higher_bound_train, color='blue', alpha=0.2, label='95% PPC interval')
    # Plot blue bands for GP test 95% PPC interval
    plt.fill_between(X_test, CI_lower_bound_test, CI_higher_bound_test, color='blue', alpha=0.2)
    # Plot dark blue bands for GP train uncertainty
    plt.fill_between(X_train, y_gpr_train - y_std_train, y_gpr_train + y_std_train, color='blue', alpha=0.4, label='GP stdev')
    # Plot dark blue bands for GP test uncertainty
    plt.fill_between(X_test, y_gpr_test - y_std_test, y_gpr_test + y_std_test, color='blue', alpha=0.4)

    # Add x/y labels and plot title
    plt.xlabel('Time (Weekly)')
    plt.ylabel('Average Avocado Price (USD)')
    plt.title(f'GPR: {region} {type_.title()} Avocado Price')

    # Convert pd.datetime to string datetime
    xticks = list(df.index.strftime('%m-%d-%Y'))
    # Only show every 10th tick to avoid overcrowding
    for i in range(len(xticks)):
        if i % 10:
            xticks[i] = ''

    # Join train/test data and label the xtick marks with the date strings
    plt.xticks(np.concatenate((X_train, X_test)), labels=xticks)

    # Rotates x axis date labels by 45 degrees
    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(45)

    # Add legend to plot
    plt.legend(loc='upper left', scatterpoints=1, prop={'size': 8})
    # Add lightly shaded grid to plot
    plt.grid(linewidth=0.25, alpha=0.5)
    # Adjust plot spacing
    plt.subplots_adjust(bottom=0.22)

    # Save plot to /figures with timestamp
    plt.savefig(f'figures/GPR_{time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime())}.png')

    # Show plot in interactive mode
    plt.show()


def future_trend_plot(df, gpr, y_test, y_gpr_test, y_std_test):
    """
    Parameters
    ----------
    df : pd.DataFrame
        contains datetime info

    gpr : sklearn.gaussian_process.GaussianProcessRegressor
        Stores pre-trained model

    y_test : np.ndarray
        contains test data

    yy_gpr_test : np.ndarray
        contains test GP model predictions

    y_std_test : np.ndarray
        contains test model uncertainty

    Effects
    -------
    Opens interactive plot of model predictions extending 
    several years into the future.

    """

    # Save the results for potential future analysis
    results = pd.DataFrame({'truth': y_test, 'predicted_val': y_gpr_test, "predicted_std": y_std_test}, index=df.index[len(y_train):])
    results.to_pickle("./data/regression_results.pkl")

    # Generate future X data
    more_X_train = np.arange(925, 2000, 7).reshape(-1, 1)
    # Predict for given X data
    more_y_val, more_y_std = gpr.predict(more_X_train, return_std=True)

    # Plot shape of model
    plt.plot(more_X_train, more_y_val)

    # Open interactive plot
    plt.show()
