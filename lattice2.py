import numpy as np
from bsmarket2 import YieldCurve, Stock


class BinomialTree(object):
    def __init__(self, steps):
        self.size = steps + 1
        self.tree = [np.empty(x+1, dtype=tuple) for x in range(self.size)]

    def __str__(self):
        return str(self.tree)


class GS(BinomialTree):

    def __init__(self, steps, principal, maturity, coupon_rate, coupon_freq,
                 spread, month, symbol, conversion_ratio, call):
        # inheritance
        super().__init__(steps)
        # save inputs
        self.steps = steps
        self.principal = principal
        self.maturity = maturity    # in years
        # coupon payment
        self.coupon_rate = coupon_rate
        self.coupon_freq = coupon_freq
        # credit risk
        self.spread = spread
        # identifiers
        self.month = month
        self.symbol = symbol
        # convertible feature
        self.conversion_ratio = conversion_ratio
        # call provision
        self.call = call
        # continuous dividend
        self.div_cont = 0.00
        # discrete dividend
        self.div_disc = 0.00
        self.div_freq = 1

        # objects
        self.eq = Stock(month, symbol)
        self.rf = YieldCurve(month)

        # set parameters
        self.stock = self.eq.price
        self.volatility = self.eq.vol / 100
        self.dt = self.maturity / self.steps
        self.up = np.exp(self.volatility * self.dt ** 0.5)
        self.down = 1 / self.up

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
        if step != 0 and step * self.dt % self.coupon_freq == 0:
            return self.coupon_rate * self.principal
        else:
            return 0

    # discrete dividend payment
    def __dividend(self, step):
        if step * self.dt % self.div_freq == 0:
            return self.div_disc
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
                self.tree[j][i - 1] = self.tree[j - 1][i - 1] * self.up * (1 - self.__dividend(j))
            self.tree[i][i] = self.tree[i - 1][i - 1] * self.down * (1 - self.__dividend(i))
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
        print("GS price: ", self.tree[0][0][1]/10)
        return None
