import numpy as np
from bsmarket2 import YieldCurve, Stock
from sklearn import linear_model


class MonteCarlo(object):
    def __init__(self, steps, iterations):
        # number of time steps
        self.steps = steps
        # number of iterations
        self.iterations = iterations
        # data structure
        self.size = steps + 1
        self.matrix = np.empty([2*self.iterations, self.size], dtype=float)

    def __str__(self):
        return str(self.matrix)


class LSM(MonteCarlo):
    def __init__(self, steps, its, order, principal, maturity, coupon_rate, coupon_freq,
                 dividend, default, month, symbol, conversion_ratio, call):
        # set seed
        np.random.seed(3000)
        # inheritance
        super().__init__(steps, its)
        self.order = order
        self.principal = principal
        self.maturity = maturity  # in years
        self.coupon_rate = coupon_rate
        self.coupon_freq = coupon_freq
        # continuous dividend
        self.dividend = dividend
        # credit risk
        self.def_intensity = default
        # identifiers
        self.month = month
        self.symbol = symbol
        # convertible feature
        self.conversion_ratio = conversion_ratio
        # call provision
        self.call = call

        # objects
        self.eq = Stock(month, symbol)
        self.rf = YieldCurve(month)

        # set parameters
        self.stock = self.eq.price
        self.strike = self.principal / self.conversion_ratio
        self.volatility = self.eq.vol / 100
        self.dt = self.maturity / self.steps

        # price of the bond
        self.price = None
        # random path generation
        self.__sim()
        # backward induction with Longstaff-Schwartz method
        self.__lsm()

    def get_price(self):
        return self.price

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
        if step != 0 and step * self.dt % self.coupon_freq == 0:
            return self.coupon_rate * self.principal
        else:
            return 0

    def __sim(self):
        for i in range(0, self.iterations):
            # random path
            self.matrix[i][0] = self.stock
            # antithetic path
            # self.matrix[i+1][0] = self.stock
            for t in range(1, self.size):
                # drift term with credit risk and dividend
                mu = self.rf.forward((t-1) * self.dt, t * self.dt) - \
                     self.def_intensity - \
                     self.dividend - \
                     self.volatility ** 2 / 2
                # random variable
                dw = np.sqrt(self.dt) * np.random.standard_normal()
                # random path
                self.matrix[i][t] = self.matrix[i][t-1] * np.exp(mu * self.dt + self.volatility * dw)
                # antithetic path
                # self.matrix[i+1][t] = self.matrix[i+1][t-1] * np.exp(mu * self.dt - self.volatility * dw)
        # transpose data structure for the backward induction
        self.matrix = self.matrix.transpose()
        return None

    def __lsm(self):
        # price of the bond at the different time steps
        bond = np.array([self.__payoff(stock, self.principal) + self.coupon_rate * self.principal
                         for stock in self.matrix[-1]])

        # price of the european no coupon bond
        european = np.mean(bond) * self.rf.df(self.maturity)
        print("LS european: ", european)

        for t in reversed(range(1, self.steps)):
            # in the money indicator vector at the different time steps
            itm = [bool(stock > self.strike) for stock in self.matrix[t]]

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
                # coupon payment
                bond[i] += self.__coupon(t)

        self.price = np.mean(bond) * self.rf.df(self.dt)
        print("LS american: ", np.mean(bond) * self.rf.df(self.dt))
        return None
