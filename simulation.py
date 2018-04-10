import datetime as dt
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from bsmarket import Option
from sklearn import linear_model
from scipy.stats import norm
from functools import reduce

# np.random.seed(150000)


class MonteCarlo(object):
    def __init__(self, steps, iterations):
        self.steps = steps
        self.iterations = iterations
        self.size = steps + 1
        self.matrix = np.empty([self.iterations, self.size], dtype=float)
        self.at_var = np.empty([self.iterations, self.size], dtype=float)

    def __str__(self):
        return str(self.matrix)


class Example(MonteCarlo):
    def __init__(self, stock, strike, maturity, rate, sigma):
        np.random.seed(1)
        self.steps = 4
        self.iterations = 10000
        self.size = self.steps + 1
        super().__init__(self.steps, self.iterations)
        self.stock = stock
        self.strike = strike
        self.maturity = maturity
        self.rate = rate
        self.sigma = sigma

        self.dt = self.maturity / self.steps
        self.df = np.exp(-self.rate * self.dt)

        self.stock_mx = self.__sim()
        self.am_call, self.am_put, self.eu_call, self.eu_put = self.__lsm()

    def bs_call(self):
        d1 = (np.log(self.stock/self.strike) + (self.rate + self.sigma ** 2 / 2) * self.maturity) / \
             (self.sigma * np.sqrt(self.maturity))
        d2 = d1 - self.sigma * np.sqrt(self.maturity)
        return self.stock * norm.cdf(d1) - np.exp(-self.rate * self.maturity) * self.strike * norm.cdf(d2)

    def bs_put(self):
        d1 = (np.log(self.stock / self.strike) + (self.rate + self.sigma ** 2 / 2) * self.maturity) / \
             (self.sigma * np.sqrt(self.maturity))
        d2 = d1 - self.sigma * np.sqrt(self.maturity)
        return np.exp(-self.rate * self.maturity) * self.strike * norm.cdf(-d2) - self.stock * norm.cdf(-d1)

    def forward(self):
        return self.stock - np.exp(-self.rate * self.maturity) * self.strike

    def __sim(self):
        result = np.empty([self.iterations, self.size], dtype=float)
        for i, path in enumerate(result):
            path[0] = self.stock
            for j, step in enumerate(path[1:]):
                path[j + 1] = path[j] * np.exp((self.rate - .5 * (self.sigma ** 2)) * self.dt + self.sigma *
                                               np.sqrt(self.dt) * sp.random.standard_normal())
        return result.transpose()

    def __lsm(self):
        res_put = np.maximum(self.strike - self.stock_mx[-1], 0)
        eu_put = np.mean(res_put) * np.exp(-self.rate * self.maturity)

        res_call = np.maximum(self.stock_mx[-1] - self.strike, 0)
        eu_call = np.mean(res_call) * np.exp(-self.rate * self.maturity)
        for t in range(self.steps - 1, 0, -1):
            print(t)
            order = 3
            # put option
            actual_payoff = np.maximum(self.strike - self.stock_mx[t], 0)
            x = np.array([[self.laguerre(x, r) for r in reversed(range(order))] for i, x in enumerate(self.stock_mx[t])
                          if actual_payoff[i] > 0])
            y = [y for j, y in enumerate(res_put * self.df) if actual_payoff[j] > 0]
            reg = linear_model.LinearRegression(fit_intercept=False)
            reg.fit(x, y)
            x = np.array([[self.laguerre(x, r) for r in reversed(range(order))]
                          for i, x in enumerate(self.stock_mx[t])])
            con = reg.predict(X=x)
            res_put = np.where((actual_payoff > con) & (actual_payoff > 0), actual_payoff, res_put * self.df)
            # call option
            actual_payoff = np.maximum(self.stock_mx[t] - self.strike, 0)
            x = np.array([[self.laguerre(x, r) for r in reversed(range(order))] for i, x in enumerate(self.stock_mx[t])
                          if actual_payoff[i] > 0])
            y = [y for j, y in enumerate(res_call * self.df) if actual_payoff[j] > 0]
            reg = linear_model.LinearRegression(fit_intercept=False)
            reg.fit(x, y)
            x = np.array([[self.laguerre(x, r) for r in reversed(range(order))]
                          for i, x in enumerate(self.stock_mx[t])])
            con = reg.predict(X=x)
            res_call = np.where((actual_payoff > con) & (actual_payoff > 0), actual_payoff, res_call * self.df)
        return np.average(res_call) * self.df, np.average(res_put) * self.df, eu_call, eu_put

    def payoff_function(self, stock):
        pass

    @staticmethod
    def basis(x, order):
        weight = np.exp(-x / 2)
        if order == 0:
            coefficients = [1.]
        elif order == 1:
            coefficients = [-1., 1.]
        elif order == 2:
            coefficients = [.5, -2., 1.]
        elif order == 3:
            coefficients = [-1/6, 3/2, -3, 1]
        elif order == 4:
            coefficients = [1/24, -2/3, 3, -4, 1]
        elif order == 5:
            coefficients = [-1/120, 5/24, -10/6, 5, -5, 1]
        else:
            raise AttributeError
        res = 0
        for r, c in enumerate(reversed(coefficients)):
            res += c * x ** r
        return weight * res

    # insufficient
    def legendre(self, x, order):
        if order == 0:
            return 1
        elif order == 1:
            return x
        else:
            return ((2 * order + 1) * x * self.legendre(x, order - 1) - order * self.legendre(x, order - 2)) / \
                   (order + 1)

    # insufficient
    def laguerre(self, x, order):
        if order == 0:
            return 1
        elif order == 1:
            return 1 - x
        else:
            return ((2 * order + 1 - x) * self.laguerre(x, order - 1) - order * self.laguerre(x, order - 2)) / \
                   (order + 1)


class LSM(MonteCarlo):
    def __init__(self, steps, its, order, principal, maturity, coupon, coupon_freq, ticker, ccy, con_ratio, call):
        np.random.seed(3000)
        super().__init__(steps, its)
        self.order = order
        self.principal = principal
        self.maturity = maturity  # years
        self.coupon = coupon
        self.coupon_freq = coupon_freq
        self.ticker = ticker
        self.ccy = ccy
        self.cr = con_ratio
        # call provision
        self.call = call
        # dilution
        self.stock_out = 168.07  # million
        self.cb_out = 1.2  # million
        # continuous dividend
        self.div_cont = 0.00
        # discrete dividend
        self.div_disc = 0.0
        self.div_freq = 9999999
        # credit risk
        self.spread = None
        self.def_intensity = 0.00

        self.expiry = dt.date.today() + dt.timedelta(days=365 * maturity)
        self.strike = principal / con_ratio  # conversion price?

        self.option = Option(ticker, ccy, 'call', self.expiry, self.strike)
        self.stock = self.option.stock
        self.rf = self.option.yc

        self.volatility = max(self.stock.hist_vol, self.option.implied_vol())
        self.dt = self.maturity / self.steps

        self.__sim()
        self.__lsm()

    # recursive
    def __legendre(self, x, order):
        if order == 0:
            return 1
        elif order == 1:
            return x
        else:
            return ((2 * order + 1) * x * self.__legendre(x, order - 1) - order * self.__legendre(x, order - 2)) / \
                   (order + 1)

    # recursive
    def __laguerre(self, x, order):
        if order == 0:
            return 1
        elif order == 1:
            return 1 - x
        else:
            return ((2 * order + 1 - x) * self.__laguerre(x, order - 1) - order * self.__laguerre(x, order - 2)) / \
                   (order + 1)

    def __diluted(self, stock):
        return (self.stock_out * stock + self.cb_out * self.principal) / (self.stock_out + self.cb_out * self.cr)

    def __payoff(self, stock, cont):
        return max(min(self.call, cont), self.cr * self.__diluted(stock))

    # coupon payment
    def __coupon(self, step):
        if step != 0 and step * self.dt % self.coupon_freq == 0:
            return self.coupon * self.principal
        else:
            return 0

    def __sim(self):
        for i in range(0, self.iterations, 2):
            # random path
            self.matrix[i][0] = self.stock.price
            # antithetic
            self.matrix[i+1][0] = self.stock.price
            for t in range(1, self.size):
                alpha = self.rf.forward((t-1) * self.dt, t * self.dt) - self.def_intensity - \
                        self.div_cont - self.volatility ** 2 / 2
                dw = np.sqrt(self.dt) * np.random.standard_normal()
                # random path
                self.matrix[i][t] = self.matrix[i][t-1] * np.exp(alpha * self.dt + self.volatility * dw)
                # antithetic
                self.matrix[i+1][t] = self.matrix[i+1][t-1] * np.exp(alpha * self.dt - self.volatility * dw)
        self.matrix = self.matrix.transpose()
        return None

    def __lsm(self):
        # price of the bond at the different time steps
        bond = np.array([self.__payoff(stock, self.principal) + self.coupon * self.principal
                         for stock in self.matrix[-1]])

        # price of the european no coupon bond
        european = np.mean(bond) * self.rf.df(self.maturity)
        print(european)

        for t in reversed(range(1, self.steps)):
            # in the money indicator vector at the different time steps
            itm = [bool(self.__diluted(stock) > self.strike) for stock in self.matrix[t]]

            # matrix* of stock prices where the bond is itm
            stock_itm = np.array([[self.__laguerre(stock, r) for r in reversed(range(self.order))]
                                  for i, stock in enumerate(self.matrix[t]) if itm[i]])

            # vector of the discounted itm bond prices
            bond_itm = np.array([y for j, y in enumerate(bond * self.rf.df((t+1) * self.dt, t * self.dt)) if itm[j]])

            # linear regression
            model = linear_model.LinearRegression(fit_intercept=False)
            model.fit(X=stock_itm, y=bond_itm)

            # matrix* of all stock prices
            stock = np.array([[self.__laguerre(stock, r) for r in reversed(range(self.order))]
                              for i, stock in enumerate(self.matrix[t])])

            # estimated continuation value
            continuation_value = model.predict(X=stock)

            # stock_itm = np.array([x for i, x in enumerate(self.matrix[t]) if itm[i]])
            # stock = np.array([x for x in self.matrix[t]])
            # plt.plot(stock_itm, y, 'o', stock, con, '-')
            # plt.show()

            for i, cont in enumerate(continuation_value):
                if self.__payoff(self.matrix[t][i], cont) == cont:
                    bond[i] *= self.rf.df((t + 1) * self.dt, t * self.dt)
                else:
                    bond[i] = self.__payoff(self.matrix[t][i], cont)
                bond[i] += self.__coupon(t)

        print(np.mean(bond) * self.rf.df(self.dt))
        return bond, european
