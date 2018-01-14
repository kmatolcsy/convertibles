from bsmarket import YieldCurve, Stock, Option, Bond
import datetime as dt
import math


def test_yc():
    yc_eur = YieldCurve("EUR", "dec")
    yc_eur.show()
    print(yc_eur)
    for x in range(10):
        print(yc_eur.spot(x))
        print(yc_eur.forward(x, x+1))
        print(yc_eur.df(x))
        print(yc_eur.af(x))

    yc_usd = YieldCurve("USD", "dec")
    yc_usd.show()
    print(yc_usd)
    for x in range(10):
        print(yc_usd.spot(x))
        print(yc_usd.forward(x, x+1))
        print(yc_usd.df(x))
        print(yc_usd.af(x))

    yc_eur = YieldCurve("EUR", "inc")
    yc_eur.show()
    print(yc_eur)
    for x in range(10):
        print(yc_eur.spot(x))
        print(yc_eur.forward(x, x+1))
        print(yc_eur.df(x))
        print(yc_eur.af(x))

    yc_usd = YieldCurve("USD", "inc")
    yc_usd.show()
    print(yc_usd)
    for x in range(10):
        print(yc_usd.spot(x))
        print(yc_usd.forward(x, x+1))
        print(yc_usd.df(x))
        print(yc_usd.af(x))


def test_stock():
    apple = Stock("AAPL", stress="type1")
    print(apple.price)
    print(apple.hist_vol)
    assert 0 < apple.hist_vol < 1


def test_option():
    apple = Option("AAPL", "USD", option_type="put", expiry=dt.date(2019, 1, 1), strike=180)
    print("stock price " + str(apple.stock.price))
    print("hist vol " + str(apple.stock.hist_vol))
    print("strike " + str(apple.strike))
    print("maturity " + str(apple.maturity))
    print("option price " + str(apple.price))
    print("impl vol " + str(apple.implied_vol()))
    assert apple.stock.hist_vol < apple.implied_vol()


def test_bond():
    yc = YieldCurve("USD")
    coupon = math.exp(yc.spot(1)) - 1
    a = Bond(principal=1000,
             maturity=1,
             coupon=coupon,
             freq=1,
             yield_curve=yc)
    print(a.value)
    assert round(a.value, 4) == 1000.
    print(a.duration())
    assert a.duration() == 1
    print(a.modified_duration())

    coupon = math.exp(yc.spot(3) * 3) - 1
    b = Bond(principal=1000,
             maturity=3,
             coupon=coupon,
             freq=3,
             yield_curve=yc)
    print(b.value)
    assert round(b.value, 4) == 1000.
    print(b.duration())
    assert b.duration() == 3
    print(b.modified_duration())

    coupon = 0.0207813716493259
    b = Bond(principal=1000,
             maturity=3,
             coupon=coupon,
             freq=1,
             yield_curve=yc)
    print(b.value)
    # assert int(b.price) == 1000
    print(b.duration())
    print(b.modified_duration())


if __name__ == "__main__":
    test_yc()
    test_stock()
    test_option()
    test_bond()
    print("end of test")
