import numpy as np
import pandas as pd
import datetime as dt
from dateutil import relativedelta
from bsmarket3 import YieldCurve, Stock


class BinomialTree(object):
    def __init__(self, steps):
        self.size = steps + 1
        self.tree = [np.empty(x+1, dtype=tuple) for x in range(self.size)]

    def __str__(self):
        return str(self.tree)


class GS(BinomialTree):

    def __init__(self, spread, month, symbol, ir_stress=None, eq_stress=None):
        # save inputs
        self.month = month
        self.symbol = symbol

        # credit risk
        self.spread = spread

        # read bond data
        df = pd.read_excel("data/PARAMETERS.xls", index_col=0, header=0)
        df = df.transpose()

        # set parameters
        self.principal = int(df["Principal"][symbol])
        # coupon payment
        self.coupon_rate = float(df["Coupon_rate"][symbol])
        self.coupon_freq = float(df["Coupon_freq"][symbol])
        # absolute maturity and valuation date
        absolute_maturity = df["Maturity"][symbol].date()
        # year and month
        m = month % 12 + 1
        y = 2017 + month // 12
        valuation_date = dt.date(y , m, 1) - dt.timedelta(days=1)
        # maturity
        self.maturity = (absolute_maturity - valuation_date).days / 365
        # print("mat: ", self.maturity)
        # continuous dividend
        self.div_cont = float(df["Dividend_rate"][symbol])
        # convertible feature
        self.conversion_ratio = float(df["Conversion_ratio"][symbol])
        # call provision
        self.call = None if df["Call_price"][symbol] == 0 else df["Call_price"][symbol]

        # objects
        self.rf = YieldCurve(month, stress=ir_stress)
        self.eq = Stock(month, symbol, stress=eq_stress)

        # set parameters
        self.stock = self.eq.price
        self.volatility = self.eq.vol
        difference = relativedelta.relativedelta(absolute_maturity, valuation_date)
        self.steps = difference.years * 12 + difference.months
        # print("steps: ", self.steps)
        self.dt = self.maturity / self.steps
        self.up = np.exp(self.volatility * self.dt ** 0.5)
        self.down = 1 / self.up

        # inheritance
        super().__init__(self.steps)
        # price of the bond
        self.price = None
        # build stock tree
        self.__build_stock()
        # build option tree
        self.__build_derivative()

    def get_price(self):
        return self.price

    # risk neutral probability
    def __prob(self, step):
        f = self.rf.forward(step * self.dt, (step + 1) * self.dt)
        return (np.exp(f * self.dt) - self.down) / (self.up - self.down)

    # coupon payment
    def __coupon(self, step):
        if (self.steps - step) % (self.coupon_freq * 12) == 0:
            return self.coupon_rate * self.principal
        else:
            return 0

    # credit spread at final nodes
    def __spread(self, stock):
        if self.principal > self.conversion_ratio * stock:
            return self.spread
        else:
            return 0

    # payoff function
    def __payoff(self, stock, roll):
        if self.call is None:
            return max(roll, self.conversion_ratio * stock)
        else:
            return max(min(self.call, roll), self.conversion_ratio * stock)

    def __build_stock(self):
        self.tree[0][0] = self.stock
        for i in range(1, self.size):
            for j in range(i, self.size):
                self.tree[j][i - 1] = self.tree[j - 1][i - 1] * self.up
            self.tree[i][i] = self.tree[i - 1][i - 1] * self.down
        return None

    def __build_derivative(self):
        for i in range(self.size):
            self.tree[-1][i] = (self.tree[-1][i],
                                self.__payoff(self.tree[-1][i], self.principal) + self.coupon_rate * self.principal,
                                self.__spread(self.tree[-1][i]))
        for j in reversed(range(self.size-1)):
            p = self.__prob(j)
            c = self.__coupon(j)
            for i in range(j+1):
                cs = p * self.tree[j+1][i][2] + (1-p) * self.tree[j+1][i+1][2]
                df = np.exp(-(self.rf.forward(j*self.dt, (j+1)*self.dt) + cs) * self.dt)
                roll = df * (p * self.tree[j+1][i][1] + (1-p) * self.tree[j+1][i+1][1])
                self.tree[j][i] = (self.tree[j][i],
                                   self.__payoff(self.tree[j][i], roll) + c,
                                   cs)
        self.price = self.tree[0][0][1]
        # print("GS price: ", self.tree[0][0][1]/10)
        return None
