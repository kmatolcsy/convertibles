import datetime as dt
import numpy as np
from scipy.stats import norm
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from matplotlib import style
import quandl
import ecb

style.use("ggplot")
quandl.ApiConfig.api_key = "MGThycVq2kfvk_VR1Jn_"


class YieldCurve(object):
    __stress_increase = np.array(  # Article 166
        [.7, .7, .7, .64, .59, .55, .52, .49, .47, .44, .42, .39, .37, .35, .34, .33, .31, .3, .29, .27, .26, .2])
    __stress_decrease = np.array(  # Article 167
        [.75, .75, .65, .56, .5, .46, .42, .39, .36, .33, .31, .3, .29, .28, .28, .27, .28, .28, .28, .29, .29, .2])
    __stress_tenors = np.array([x for x in range(21)] + [90])

    def __init__(self, ccy, stress=None):
        # save inputs
        self.date = dt.date.today()     # quandl not updated on a daily basis
        self.ccy = ccy
        self.stress = stress

        # attributes
        self.tenors, self.rates, self.data = self.__set_ccy()

    def __str__(self):
        return str(self.data)

    def __set_ccy(self):
        tenors = None
        rates = None
        data = None
        if self.ccy == "USD":
            fail = True
            while fail:
                try:
                    data = quandl.get("USTREASURY/YIELD", start_date=self.date, end_date=self.date)
                    rates = data.values[0]  # get rates from data
                    fail = False
                except Exception as e:
                    print("US Treasury rates on date ", str(self.date), "are not available \t", str(e))
                    self.date -= dt.timedelta(days=1)
                finally:
                    tenors = np.array([1/12, 1/4, 1/2, 1, 2, 3, 5, 7, 10, 20, 30])
        elif self.ccy == "EUR":
            tenors = np.array([1 / 4, 1 / 2, 3 / 4] + [x for x in range(1, 31)])  # build tenors
            rates = ecb.Rates().get()  # get rates from ECB web
            data = dict(zip(tenors, rates))  # build data
        else:
            print("""unknown currency""")
            raise AttributeError
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

    def af(self, maturity, base=0, freq=1):
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

    def __init__(self, ticker, stress=None):

        self.ticker = ticker
        self.stress = stress
        # attributes
        self.start = str(dt.date.today() - dt.timedelta(days=365))
        self.end = str(dt.date.today())
        self.data = self.__get_data()       # get data from Yahoo finance
        self.price = self.__get_price()       # get price from data
        self.hist_vol = self.__get_hist()

    def __str__(self):
        return str(self.data)

    @staticmethod
    def __symmetric_adj():
        start = str(dt.date.today() - dt.timedelta(days=37))
        end = str(dt.date.today() - dt.timedelta(days=1))
        tickers = np.array(
            ['^AEX', '^FCHI', '^GDAXI', '^FTAS', 'FTSEMIB.MI', '^IBEX', '^SSMI', '^GSPC', '^OMX', '^N225'])
        weights = np.array([.14, .14, .14, .14, .08, .08, .02, .08, .08, .02])
        # get index data
        index = []
        for ticker in tickers:
            fail = True
            attempt = 1
            while fail:
                try:
                    index.append(web.DataReader(ticker, 'yahoo', start, end)['Adj Close'])
                    fail = False
                except Exception as e:
                    print(str(ticker), str(attempt), "attempt failed \t", str(e))
                    attempt += 1
        # calculate current index (CI) value
        ci = 0
        for x, w in zip(index, weights):
            ci += w * x[-1]
        # calculate average index (AI) value
        ai = 0
        for x, w in zip(index, weights):
            ai += w * np.mean(x)
        # calculate symmetric adjustment
        symmetric_adj = 0.5 * ((ci - ai) / ai - .08)
        return max(-.1, min(symmetric_adj, .1))

    def __get_data(self):
        table = None
        fail = True
        attempt = 1
        while fail:
            try:
                table = web.DataReader(self.ticker, 'yahoo', self.start, self.end)
                fail = False
            except Exception as e:
                print(str(self.ticker), str(attempt), "attempt failed \t", str(e))
        return table['Adj Close']

    def __get_price(self):
        if self.stress == "type1":
            return float(self.data.values[-1]) * (1 - self.__stress_1 - self.__symmetric_adj())
        elif self.stress == "type2":
            return float(self.data.values[-1]) * (1 - self.__stress_2 - self.__symmetric_adj())
        else:
            return float(self.data.values[-1])

    def __get_hist(self):
        log_returns = np.array([np.log(self.data.values[-x]/self.data.values[-x-1]) for x in range(1, len(self.data))])
        return np.std(log_returns) * 252 ** 0.5


class Option(object):

    def __init__(self, ticker, ccy, option_type='call', expiry=None, strike=None):
        # objects
        self.stock = Stock(ticker)
        self.option = web.Options(ticker, "yahoo")
        self.yc = YieldCurve(ccy)

        # save inputs
        self.ticker = ticker
        self.ccy = ccy
        self.option_type = option_type
        self.expiry = expiry    # modified in __get_price()
        self.strike = strike    # modified in __get_price()

        # attributes
        self.maturity = None
        self.data = None
        self.price = self.__get_price()

    def __str__(self):
        return str(self.data)

    def __expiry_market(self):
        if self.expiry is None or self.expiry > self.option.expiry_dates[-1]:
            return self.option.expiry_dates[-1]
        for date in self.option.expiry_dates:
            if self.expiry <= date:
                return date

    def __strike_market(self, table):
        if self.strike is None:
            self.strike = self.stock.price
        for x, y in table.index.values:
            if self.strike <= x:
                return x

    def __get_price(self):
        expiry = self.__expiry_market()
        table = self.option.get_all_data().xs((expiry, self.option_type), level=('Expiry', 'Type'), drop_level=True)
        strike = self.__strike_market(table)
        self.data = table['Last']
        self.maturity = (expiry - dt.date.today()) / dt.timedelta(days=365)
        # update inputs
        self.strike = strike
        self.expiry = expiry
        return float(self.data[strike])

    def __bs_diff(self, sigma):
        d1 = (np.log(self.stock.price/self.strike) + (self.yc.spot(self.maturity) + sigma ** 2 / 2) * self.maturity) / \
             (sigma * self.maturity ** 0.5)
        d2 = d1 - sigma * self.maturity ** 0.5
        return self.stock.price * norm.cdf(d1) - self.strike * self.yc.df(self.maturity) * norm.cdf(d2) - self.price

    def __vega(self, sigma):
        d1 = (np.log(self.stock.price/self.strike) + (self.yc.spot(self.maturity) + sigma ** 2 / 2) * self.maturity) / \
             (sigma * self.maturity ** 0.5)
        return self.stock.price * norm.pdf(d1) * self.maturity ** 0.5

    def implied_vol(self, epsilon=0.0001):
        guess = self.stock.hist_vol
        delta = abs(0 - self.__bs_diff(guess))
        while delta > epsilon:
            guess -= self.__bs_diff(guess) / self.__vega(guess)
            delta = abs(0 - self.__bs_diff(guess))
        return guess


class Bond(object):
    __a = np.array([
        np.array([.0, .045, .07, .095, .12]),
        np.array([.0, .055, .084, .109, .134]),
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
        np.array([.03, .017, .012, .012, .005]),
    ])

    def __init__(self, principal, maturity, coupon, freq, yield_curve, stress=False, quality=None):
        # save inputs
        self.principal = principal
        self.maturity = maturity
        self.coupon = coupon    # coupon rate
        self.freq = freq        # number of coupon payments in a year
        self.yc = yield_curve
        self.stress = stress
        self.quality = quality

        # private attribute for background calculation
        self.__value = (yield_curve.af(maturity, 0, freq) * coupon + yield_curve.df(maturity)) * principal

        # attributes
        self.value = self.__set_value()

    @staticmethod
    def __range(end, start, step):
        if start < end and step > 0:
            current = end
            while current > start:
                yield current
                current -= step

    def __stress(self, duration):
        assert duration > 0
        quality_idx = self.quality
        duration_idx = 4
        x = duration - 20
        if self.quality is None:
            quality_idx = 6
        for i in range(0, 4):
            low = i * 5
            high = low + 5
            if low < duration <= high:
                duration_idx = i
                x = duration - low
        result = self.__a[quality_idx][duration_idx] + self.__b[quality_idx][duration_idx] * x
        return min(result, 1)

    def __set_value(self):
        if self.stress:
            return self.__value * (1 - self.__stress(self.modified_duration()))
        else:
            return self.__value

    def __ytm_diff(self, ytm):
        s = self.principal / (1 + ytm) ** self.maturity
        for t in self.__range(self.maturity, 0, self.freq):
            s += self.coupon * self.principal / (1 + ytm) ** t
        return s - self.__value

    def __ytm_derivative(self, ytm):
        s = - self.principal * self.maturity * (1 + ytm) ** (- self.maturity - 1)
        for t in self.__range(self.maturity, 0, self.freq):
            s += - self.coupon * self.principal * t * (1 + ytm) ** (- t - 1)
        return s

    def ytm(self, epsilon=0.001):
        guess = (self.coupon * self.principal + (self.principal - self.__value) / int(self.maturity / self.freq)) / \
                ((self.principal + self.__value) / 2)
        delta = abs(0 - self.__ytm_diff(guess))
        while delta > epsilon:
            guess -= self.__ytm_diff(guess) / self.__ytm_derivative(guess)
            delta = abs(0 - self.__ytm_diff(guess))
        return guess

    def duration(self):
        s = self.maturity * self.principal * self.yc.df(self.maturity)
        for t in self.__range(self.maturity, 0, self.freq):
            s += t * self.coupon * self.principal * self.yc.df(t)
        return s / self.__value

    def modified_duration(self):
        return self.duration()      # continuously compounded yields
