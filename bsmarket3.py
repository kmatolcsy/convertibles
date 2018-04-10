import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")


class YieldCurve(object):
    __stress_increase = np.array(  # Article 166
        [.7, .7, .7, .64, .59, .55, .52, .49, .47, .44, .42, .39, .37, .35, .34, .33, .31, .3, .29, .27, .26, .2])
    __stress_decrease = np.array(  # Article 167
        [.75, .75, .65, .56, .5, .46, .42, .39, .36, .33, .31, .3, .29, .28, .28, .27, .28, .28, .28, .29, .29, .2])
    __stress_tenors = np.array([x for x in range(21)] + [90])

    def __init__(self, month, stress=None):
        # save inputs
        self.date = dt.date.today()
        self.month = month
        self.stress = stress

        # attributes
        self.tenors, self.rates, self.data = self.__set_month()

    def __str__(self):
        return str(self.data)

    def __set_month(self):
        df = pd.read_excel("data/USTREASURY.xls", index_col=0, header=0)
        df = df.transpose()
        tenors = [1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]
        rates = list(df[self.month])
        data = dict(zip(tenors, rates))
        return tenors, rates, data

    # bisection search method in elements
    @staticmethod
    def __in(item, x):
        if x[0] <= item <= x[-1]:
            mn = 0
            mx = len(x) - 1
            j = 0
            if x[mn] == item:
                return mn
            if x[mx] == item:
                return mx
            while j <= len(x):
                if j == len(x):
                    print(""""__in function failed in YieldCurve()""")
                    raise Exception
                md = int((mx + mn) / 2)
                if md == mn or md == mx:
                    return None
                if x[md] == item:
                    return md
                elif x[md] > item:
                    mx = md
                    j += 1
                else:
                    mn = md
                    j += 1
        else:
            return None

    # bisection search method within elements
    @staticmethod
    def __within(item, x):
        if x[0] < item < x[-1]:
            mn = 0
            mx = len(x) - 1
            j = 0
            while j <= len(x):
                if j == len(x):
                    print(""""__within function failed in YieldCurve()""")
                    raise Exception
                md = int((mx + mn) / 2)
                if x[md] >= item:
                    mx = md
                    j += 1
                elif x[md + 1] < item:
                    mn = md + 1
                    j += 1
                else:
                    return md + 1
        else:
            print("{} not found within {}".format(item, x))
            return None

    # range for af
    @staticmethod
    def __range(end, start, step):
        if start < end and step > 0:
            current = end
            while current > start:
                yield current
                current -= step

    def __approx(self, item, x, y):
        if x[0] < item < x[-1]:
            k = self.__within(item, x)
            return y[k - 1] + (item - x[k - 1]) * (y[k] - y[k - 1]) / (x[k] - x[k - 1])
        elif 0 < item < x[0]:
            return item * y[0] / x[0]
        elif item <= 0:
            return 0
        else:
            return x[-1]

    def __spot(self, maturity):
        k = self.__in(maturity, self.tenors)
        if isinstance(k, int):
            return self.rates[k] / 100
        else:
            return self.__approx(maturity, self.tenors, self.rates) / 100

    def __stress(self, maturity):
        if self.stress == "inc":
            m = 1 + self.__approx(maturity, self.__stress_tenors, self.__stress_increase)
            return max(self.__spot(maturity) * m, self.__spot(maturity) + 0.01)
        elif self.stress == "dec":
            m = 1 - self.__approx(maturity, self.__stress_tenors, self.__stress_decrease)
            if self.__spot(maturity) > 0:
                return self.__spot(maturity) * m
            else:
                return self.__spot(maturity)
        else:
            print("""unknown stress""")
            raise AttributeError

    def spot(self, maturity):
        if self.stress is None:
            return self.__spot(maturity)
        else:
            return self.__stress(maturity)

    def forward(self, closer, further):
        return (self.spot(further) * further - self.spot(closer) * closer) / (further - closer)

    def df(self, maturity, base=0):
        if maturity == 0:
            return 1
        else:
            return np.exp(-self.forward(base, maturity) * (maturity - base))

    def af(self, maturity, base=0, freq=1.0):
        res = 0
        for x in self.__range(maturity, base, freq):
            res += self.df(x, base)
        return res

    def show(self):
        tenors = np.array([0, 1 / 12, 1 / 4, 1 / 2, 3 / 4] + [x for x in range(1, 31)])
        rates = np.array([self.__spot(x) for x in tenors])
        if self.stress is None:
            plt.plot(tenors, rates)
        else:
            rates_stress = np.array([self.__stress(x) for x in tenors])
            plt.plot(tenors, rates, '--', tenors, rates_stress)
        plt.show()
        return None


class Stock(object):
    __stress_1 = .39
    __stress_2 = .49

    def __init__(self, month, symbol, stress=None):
        self.month = month
        self.symbol = symbol
        self.stress = stress
        # attributes
        self.price = self.__get_price()
        self.vol = self.__get_vol()
        self.charge_pct = self.__stress_1 + self.__symmetric_adj() if stress == "type1" \
            else self.__stress_2 + self.__symmetric_adj()

    def __symmetric_adj(self):
        df = pd.read_excel("data/ADJUSTMENT.xls", index_col=0, header=0)
        df = df.transpose()
        return float(df[self.month][0])

    def __get_price(self):
        df = pd.read_excel("data/STOCK_PRICE.xls", index_col=0, header=0)
        df = df.transpose()
        if self.stress == "type1":
            return df[self.month][self.symbol] * (1 - self.__stress_1 - self.__symmetric_adj())
        elif self.stress == "type2":
            return df[self.month][self.symbol] * (1 - self.__stress_2 - self.__symmetric_adj())
        else:
            return df[self.month][self.symbol]

    def __get_vol(self):
        df = pd.read_excel("data/STOCK_VOLS.xls", index_col=0, header=0)
        df = df.transpose()
        return df[self.month][self.symbol] / 100


class Bond(object):
    __a = np.array([
        np.array([.0, .045, .07, .095, .12]),
        np.array([.0, .055, .085, .11, .135]),
        np.array([.0, .07, .105, .13, .155]),
        np.array([.0, .125, .2, .25, .3]),
        np.array([.0, .225, .35, .44, .465]),
        np.array([.0, .375, .585, .61, .635]),
        np.array([.0, .15, .235, .235, .355])
    ])
    __b = np.array([
        np.array([.009, .005, .005, .005, .005]),
        np.array([.011, .006, .005, .005, .005]),
        np.array([.014, .007, .005, .005, .005]),
        np.array([.025, .015, .01, .01, .005]),
        np.array([.045, .025, .018, .005, .005]),
        np.array([.075, .042, .005, .005, .005]),
        np.array([.03, .017, .012, .012, .005])
    ])

    def __init__(self, month, symbol, rating=None):
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

        # read bond data
        df = pd.read_excel("data/CONVERTIBLE.xls", index_col=0, header=0)
        df = df.transpose()

        self.market_price = df[month][symbol]

        # save inputs
        self.month = month
        self.symbol = symbol
        self.rating = rating

        # objects
        self.yc = YieldCurve(month)

        # attributes
        self.value = (self.yc.af(self.maturity, 0, self.coupon_freq) * self.coupon_rate +
                      self.yc.df(self.maturity)) * self.principal
        self.duration = self.dur()
        self.charge = self.market_price * self.stress_factor() * 10

    @staticmethod
    def __range(end, start, step):
        if start < end and step > 0:
            current = end
            while current > start:
                yield current
                current -= step

    def __ytm_diff(self, ytm):
        s = self.principal / (1 + ytm) ** self.maturity
        for t in self.__range(self.maturity, 0, self.coupon_freq):
            s += self.coupon_rate * self.principal / (1 + ytm) ** t
        return s - self.value

    def __ytm_derivative(self, ytm):
        s = - self.principal * self.maturity * (1 + ytm) ** (- self.maturity - 1)
        for t in self.__range(self.maturity, 0, self.coupon_freq):
            s += - self.coupon_rate * self.principal * t * (1 + ytm) ** (- t - 1)
        return s

    def ytm(self, epsilon=0.001):
        guess = (self.coupon_rate * self.principal + (self.principal - self.value) /
                 int(self.maturity / self.coupon_freq)) / ((self.principal + self.value) / 2)
        delta = abs(0 - self.__ytm_diff(guess))
        while delta > epsilon:
            guess -= self.__ytm_diff(guess) / self.__ytm_derivative(guess)
            delta = abs(0 - self.__ytm_diff(guess))
        return guess

    def dur(self):
        s = self.maturity * self.principal * self.yc.df(self.maturity)
        for t in self.__range(self.maturity, 0, self.coupon_freq):
            s += t * self.coupon_rate * self.principal * self.yc.df(t)
        return s / self.value

    def stress_factor(self):
        assert self.duration > 0
        # declare indices and x
        quality_idx = self.rating
        duration_idx = 4
        x = self.duration - 20
        # define indices and x
        if self.rating is None:
            quality_idx = 6
        for i in range(0, 4):
            low = i * 5
            high = low + 5
            if low < self.duration <= high:
                duration_idx = i
                x = self.duration - low
        result = self.__a[quality_idx][duration_idx] + self.__b[quality_idx][duration_idx] * x
        return min(result, 1)
