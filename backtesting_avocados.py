from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import pickle
import pandas as pd
from backtesting import Strategy
from backtesting import Backtest

regression_results = pickle.load(open('./data/regression_results.pkl', 'rb'))
# next predict - current predict
regression_results["signal"] = regression_results["predicted_val"] - regression_results["predicted_val"].shift()
# cerebro = bt.Cerebro()
#
# print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
#
# cerebro.run()
#
# print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
prices = pd.read_csv("./data/results.csv")
prices["High"] = prices["Open"]
prices["Low"] = prices["Open"]
prices["Close"] = prices["Open"]

class SmaCross(Strategy):
    # Define the two MA lags as *class variables*
    # for later optimization
    day = 0

    def init(self):
        # Precompute two moving averages
        pass

    def next(self):
        # If sma1 crosses above sma2, buy the asset
        if (regression_results["signal"].iloc[self.day] > 0):
            self.buy()
        else:
            self.sell()

bt = Backtest(prices, SmaCross, cash=10000)
bt.run()
bt.plot()