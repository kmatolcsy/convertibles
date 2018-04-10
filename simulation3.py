import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from dateutil import relativedelta
from bsmarket3 import YieldCurve, Stock
from sklearn import linear_model
from scipy.stats import norm


class MonteCarlo(object):
    def __init__(self, steps, iterations):
        # number of time steps
        self.steps = steps
        # number of iterations
        self.iterations = iterations
        # data structure
        self.size = steps + 1
        self.matrix = np.empty([self.iterations, self.size], dtype=float)
        self.at_var = np.empty([self.iterations, self.size], dtype=float)

    def __str__(self):
        return str(self.matrix)


class LSM(MonteCarlo):
    def __init__(self, iterations, order, default, month, symbol, ir_stress=None, eq_stress=None):
        # set seed
        np.random.seed(1)
        self.order = order
        # credit risk
        self.def_intensity = default
        # identifiers
        self.month = month
        self.symbol = symbol

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
        valuation_date = dt.date(y, m, 1) - dt.timedelta(days=1)
        # maturity
        self.maturity = (absolute_maturity - valuation_date).days / 365
        # print("mat: ", self.maturity)
        # continuous dividend
        self.dividend = float(df["Dividend_rate"][symbol])
        # convertible feature
        self.conversion_ratio = float(df["Conversion_ratio"][symbol])
        # call provision
        self.call = None if df["Call_price"][symbol] == 0 else df["Call_price"][symbol]

        # objects
        self.rf = YieldCurve(month, stress=ir_stress)
        self.eq = Stock(month, symbol, stress=eq_stress)

        # set parameters
        self.stock = self.eq.price
        self.strike = self.principal / self.conversion_ratio
        self.volatility = self.eq.vol
        difference = relativedelta.relativedelta(absolute_maturity, valuation_date)
        self.steps = difference.years * 12 + difference.months
        # print("steps: ", self.steps)
        self.dt = self.maturity / self.steps

        # inheritance
        super().__init__(self.steps, iterations)
        # price of the bond
        self.price = None
        # random path generation
        self.__sim()
        # backward induction with Longstaff-Schwartz method
        self.__lsm()

    def get_price(self):
        return self.price

    def bs_price(self):
        d1 = (np.log(1) + (self.rf.spot(self.maturity) + self.volatility ** 2 / 2) * self.maturity) / \
             (self.volatility * self.maturity ** 0.5)
        d2 = d1 - self.volatility * self.maturity ** 0.5
        return self.stock * norm.cdf(d1) - self.stock * self.rf.df(self.maturity) * norm.cdf(d2)

    # recursive definition of Legendre polynomials
    def __legendre(self, x, order):
        if order == 0:
            return 1
        elif order == 1:
            return x
        else:
            return ((2 * order + 1) * x * self.__legendre(x, order - 1) - order * self.__legendre(x, order - 2)) / \
                   (order + 1)

    # recursive definition of Laguerre polynomials
    def __laguerre(self, x, order):
        if order == 0:
            return 1
        elif order == 1:
            return 1 - x
        else:
            return ((2 * order + 1 - x) * self.__laguerre(x, order - 1) - order * self.__laguerre(x, order - 2)) / \
                   (order + 1)

    # payoff function
    def __payoff(self, stock, cont):
        if self.call is None:
            return max(cont, self.conversion_ratio * stock)
        else:
            return max(min(self.call, cont), self.conversion_ratio * stock)

    # coupon payment
    def __coupon(self, step):
        if (self.steps - step) % (self.coupon_freq * 12) == 0:
            return self.coupon_rate * self.principal
        else:
            return 0

    def __sim(self):
        for i in range(0, self.iterations):
            # random path
            self.matrix[i][0] = self.stock
            # antithetic path
            self.at_var[i][0] = self.stock
            for t in range(1, self.size):
                # drift term with credit risk and dividend
                mu = self.rf.forward((t-1) * self.dt, t * self.dt) - \
                     self.def_intensity - \
                     self.dividend - \
                     self.volatility ** 2 / 2
                # random variable
                dw = np.sqrt(self.dt) * np.random.standard_normal()
                # random path
                self.matrix[i][t] = self.matrix[i][t - 1] * np.exp(mu * self.dt + self.volatility * dw)
                # antithetic path
                self.at_var[i][t] = self.matrix[i][t - 1] * np.exp(mu * self.dt - self.volatility * dw)
        self.matrix = np.concatenate((self.matrix, self.at_var))
        # transpose data structure for the backward induction
        self.matrix = self.matrix.transpose()
        return None

    def __lsm(self):
        # price of the bond at the different time steps
        bond = np.array([self.__payoff(stock, self.principal) + self.coupon_rate * self.principal
                         for stock in self.matrix[-1]])
        call = np.array([max(stock - self.stock, 0) for stock in self.matrix[-1]])

        # price of the european call
        # print("MC european call: ", np.mean(call) * self.rf.df(self.maturity), " --> ", self.bs_price())

        # price of the european no coupon bond
        # print("MC european bond: ", np.mean(bond) * self.rf.df(self.maturity))

        for t in reversed(range(1, self.steps)):
            # itm
            itm = np.array([bool(stock > self.strike) for stock in self.matrix[t]])

            # matrix* of stock prices where the bond is ITM
            stock_basis_itm = np.array([[self.__laguerre(stock, r) for r in reversed(range(self.order))]
                                        for i, stock in enumerate(self.matrix[t]) if itm[i]])

            # vector of the discounted ITM bond prices
            bond_itm = np.array([y for j, y in enumerate(bond * self.rf.df((t+1) * self.dt, t * self.dt)) if itm[j]])

            # linear regression
            model = linear_model.LinearRegression(fit_intercept=False)
            model.fit(X=stock_basis_itm, y=bond_itm)

            # matrix* of ALL stock prices
            stock_basis = np.array([[self.__laguerre(stock, r) for r in reversed(range(self.order))]
                                    for i, stock in enumerate(self.matrix[t])])

            # estimated continuation value
            continuation_value = model.predict(X=stock_basis)

            # stock = np.array([x for x in self.matrix[t]])
            # plt.plot(stock, bond, 'o', stock, continuation_value, 'o')
            # plt.show()

            for i, cont in enumerate(continuation_value):
                if self.__payoff(self.matrix[t][i], cont) != cont and itm[i]:
                    bond[i] = self.__payoff(self.matrix[t][i], cont)
                else:
                    bond[i] *= self.rf.df((t+1) * self.dt, t * self.dt)
                # coupon payment
                bond[i] += self.__coupon(t)

        self.price = np.mean(bond) * self.rf.df(self.dt)
        # print("LS american bond: ", np.mean(bond) * self.rf.df(self.dt))
        return None
