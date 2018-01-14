import datetime as dt
import numpy as np
from bsmarket import Option


class BinomialTree(object):
    def __init__(self, steps):
        self.size = steps + 1
        self.tree = [np.empty(x+1, dtype=tuple) for x in range(self.size)]

    def __str__(self):
        return str(self.tree)


class CRR(BinomialTree):

    def __init__(self, steps, ticker, ccy, option_type='call', maturity=1, strike=None):
        # pass input to parent
        super().__init__(steps)

        # save inputs
        self.steps = steps
        self.ticker = ticker
        self.ccy = ccy
        self.option_type = option_type
        self.maturity = maturity        # years
        self.strike = strike

        # objects
        expiry = dt.date.today() + dt.timedelta(days=365*maturity)
        self.option = Option(ticker, ccy, option_type, expiry, strike)
        self.stock = self.option.stock
        self.risk_free = self.option.yc

        # set parameters
        self.volatility = max(self.stock.hist_vol, self.option.implied_vol())
        print("""
            Volatility
            historical: {}
            implied: {} 
            """.format(self.stock.hist_vol, self.option.implied_vol()))
        self.dt = self.maturity / self.steps
        self.up = np.exp(self.volatility * self.dt ** 0.5)
        self.down = 1 / self.up

        # build stock tree
        self.__build_stock()

        # build option tree
        self.__build_derivative()

    def __build_stock(self):
        self.tree[0][0] = self.stock.price
        for i in range(1, self.size):
            for j in range(i, self.size):
                self.tree[j][i - 1] = self.tree[j - 1][i - 1] * self.up
            self.tree[i][i] = self.tree[i - 1][i - 1] * self.down
        return None

    def __payoff_function(self, stock):
        if self.option_type == 'call':
            return max(stock - self.strike, 0)
        elif self.option_type == 'put':
            return max(self.strike - stock, 0)
        else:
            print("Wrong option_type")
            raise Exception

    def __prob(self, step):
        f = self.risk_free.forward(step * self.dt, (step + 1) * self.dt)
        return ((1 + f) ** self.dt - self.down) / (self.up - self.down)

    def __build_derivative(self):
        for i in range(self.size):
            self.tree[-1][i] = (self.tree[-1][i], self.__payoff_function(self.tree[-1][i]))
        for j in reversed(range(self.size-1)):
            p = self.__prob(j)
            df = self.risk_free.df((j+1)*self.dt, j*self.dt)
            for i in range(j+1):
                self.tree[j][i] = (self.tree[j][i], df * (p * self.tree[j+1][i][1] + (1-p) * self.tree[j+1][i+1][1]))


class Oracle(BinomialTree):

    def __init__(self, steps, principal, maturity, coupon, coupon_freq, spread, ticker, ccy, con_ratio, call):
        super().__init__(steps)

        # save inputs
        self.steps = steps
        self.principal = principal
        self.maturity = maturity        # years
        self.coupon = coupon
        self.coupon_freq = coupon_freq
        self.ticker = ticker
        self.ccy = ccy
        self.cr = con_ratio
        # call provision
        self.call = call
        # dilution
        self.stock_out = 168.07     # million
        self.cb_out = 1.2  # million
        # continuous dividend
        self.div_cont = 0.00
        # discrete dividend
        self.div_disc = 0.03
        self.div_freq = 0.5
        # credit risk
        self.spread = spread

        # objects
        expiry = dt.date.today() + dt.timedelta(days=365*maturity)
        strike = principal / con_ratio       # conversion price?
        self.option = Option(ticker, ccy, 'call', expiry, strike)
        self.stock = self.option.stock
        self.risk_free = self.option.yc

        # set parameters
        self.volatility = max(self.stock.hist_vol, self.option.implied_vol())
        print("""
            Volatility
            historical: {}
            implied: {} 
            """.format(self.stock.hist_vol, self.option.implied_vol()))
        self.dt = self.maturity / self.steps
        self.up = np.exp(self.volatility * self.dt ** 0.5)
        self.down = 1 / self.up

        # build stock tree
        self.__build_stock()

        # build option tree
        print("option price: " + str(self.option.price))
        print("option maturity: " + str(self.option.maturity))
        self.__build_derivative()

    def __build_stock(self):
        self.tree[0][0] = self.stock.price
        for i in range(1, self.size):
            for j in range(i, self.size):
                self.tree[j][i - 1] = self.tree[j - 1][i - 1] * self.up * (1 - self.__dividend(j))
            self.tree[i][i] = self.tree[i - 1][i - 1] * self.down * (1 - self.__dividend(i))
        return None

    def __payoff_function(self, stock):
        stock_at_con = (self.stock_out * stock + self.cb_out * self.principal) / \
                       (self.stock_out + self.cb_out * self.cr)
        # conversion after or before coupon?
        return max(self.principal, self.cr * stock_at_con) + self.coupon * self.principal

    def __prob(self, step):
        f = self.risk_free.forward(step * self.dt, (step + 1) * self.dt)
        return (np.exp(f * self.dt) - self.down) / (self.up - self.down)

    def __coupon(self, step):
        if step != 0 and step * self.dt % self.coupon_freq == 0:
            return self.coupon * self.principal
        else:
            return 0

    def __dividend(self, step):
        if step * self.dt % self.div_freq == 0:
            return self.div_disc
        else:
            return 0

    @staticmethod
    def __roll_dist(distribution, p):
        distribution.append(0)
        result = []
        for i in range(len(distribution)):
            result.append(p * distribution[i] + (1 - p) * distribution[i - 1])
        return result

    # risk adjusted fin forward
    def __raf_forward(self, step, node, runner):
        stock_at_con = (self.stock_out * self.tree[-1][node+runner][0] + self.cb_out * self.principal) / \
                       (self.stock_out + self.cb_out * self.cr)
        if stock_at_con * self.cr > self.principal:
            return self.risk_free.forward(step * self.dt, (step + 1) * self.dt)
        else:
            return self.risk_free.forward(step * self.dt, (step + 1) * self.dt) + self.spread

    def __risk_adj_df(self, step, node, distribution):
        f = 0
        n = self.size - step
        for k in range(n):
            f += distribution[k] * self.__raf_forward(step, node, k)
        return np.exp(-f * self.dt)

    def __build_derivative(self):
        for i in range(self.size):
            self.tree[-1][i] = (self.tree[-1][i], self.__payoff_function(self.tree[-1][i]))
        distribution = [1]
        for j in reversed(range(self.size-1)):
            c = self.__coupon(j)
            p = self.__prob(j)
            distribution = self.__roll_dist(distribution, p)
            # df = self.risk_free.df((j + 1) * self.dt, j * self.dt)
            for i in range(j+1):
                adf = self.__risk_adj_df(j, i, distribution)
                rolling = adf * (p * self.tree[j+1][i][1] + (1-p) * self.tree[j+1][i+1][1])
                stock_at_con = (self.stock_out * self.tree[j][i] + self.cb_out * self.principal) / \
                               (self.stock_out + self.cb_out * self.cr)
                self.tree[j][i] = (self.tree[j][i], max(min(self.call, rolling), self.cr * stock_at_con) + c)
        return None


class Convertible(BinomialTree):

    def __init__(self, steps, principal, maturity, coupon, coupon_freq, spread, ticker, ccy, con_ratio, call):
        super().__init__(steps)
        # save inputs
        self.steps = steps
        self.principal = principal
        self.maturity = maturity        # years
        self.coupon = coupon
        self.coupon_freq = coupon_freq
        self.ticker = ticker
        self.ccy = ccy
        self.cr = con_ratio
        # call provision
        self.call = call
        # dilution
        self.stock_out = 168.07     # million
        self.cb_out = 1.2  # million
        # continuous dividend
        self.div_cont = 0.00
        # discrete dividend
        self.div_disc = 0.03
        self.div_freq = 0.5
        # credit risk
        self.spread = spread

        # objects
        expiry = dt.date.today() + dt.timedelta(days=365*maturity)
        strike = principal / con_ratio       # conversion price?
        self.option = Option(ticker, ccy, 'call', expiry, strike)
        self.stock = self.option.stock
        self.rf = self.option.yc

        # set parameters
        self.volatility = max(self.stock.hist_vol, self.option.implied_vol())
        self.dt = self.maturity / self.steps
        self.up = np.exp(self.volatility * self.dt ** 0.5)
        self.down = 1 / self.up

        # build stock tree
        self.__build_stock()

        # build option tree
        self.__build_derivative()
        print("option price: " + str(self.option.price))
        print("option maturity: " + str(self.option.maturity))

    # risk neutral probability
    def __prob(self, step):
        f = self.rf.forward(step * self.dt, (step + 1) * self.dt)
        return (np.exp(f * self.dt) - self.down) / (self.up - self.down)

    # coupon payment
    def __coupon(self, step):
        if step != 0 and step * self.dt % self.coupon_freq == 0:
            return self.coupon * self.principal
        else:
            return 0

    # discrete dividend payment
    def __dividend(self, step):
        if step * self.dt % self.div_freq == 0:
            return self.div_disc
        else:
            return 0

    # diluted stock price
    def __diluted(self, stock):
        return (self.stock_out * stock + self.cb_out * self.principal) / (self.stock_out + self.cb_out * self.cr)

    # credit spread at final nodes
    def __spread(self, stock):
        if self.principal > self.cr * self.__diluted(stock):
            return self.spread
        else:
            return 0

    # payoff function
    def __payoff(self, stock, roll):
        # conversion after or before coupon?
        return max(min(self.call, roll), self.cr * self.__diluted(stock))

    def __build_stock(self):
        self.tree[0][0] = self.stock.price
        for i in range(1, self.size):
            for j in range(i, self.size):
                self.tree[j][i - 1] = self.tree[j - 1][i - 1] * self.up * (1 - self.__dividend(j))
            self.tree[i][i] = self.tree[i - 1][i - 1] * self.down * (1 - self.__dividend(i))
        return None

    def __build_derivative(self):
        for i in range(self.size):
            self.tree[-1][i] = (self.tree[-1][i],
                                self.__payoff(self.tree[-1][i], self.principal) + self.coupon * self.principal,
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
        return None
