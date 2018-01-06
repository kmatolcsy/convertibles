from bsmarket import YieldCurve, Stock, Option, Bond
import datetime as dt
import math


def test_yc():
    yc_eur = YieldCurve("EUR", "dec")
    yc_eur.show()
    print(yc_eur)

    yc_usd = YieldCurve("USD", "dec")
    yc_usd.show()
    print(yc_usd)

    yc_eur = YieldCurve("EUR", "inc")
    yc_eur.show()
    print(yc_eur)

    yc_usd = YieldCurve("USD", "inc")
    yc_usd.show()
    print(yc_usd)


def test_stock():
    apple = Stock("AAPL")
    print(apple.price)
    print(apple.hist_vol)


def test_option():
    apple = Option("AAPL", "USD", option_type="put", expiry=dt.date(2019, 1, 1), strike=180)
    print(apple.stock.price)
    print(apple.stock.hist_vol)
    print(apple.expiry)
    print(apple.strike)
    print(apple.price)
    print(apple.implied_vol())


def test_bond():
    yc = YieldCurve("USD")
    coupon = math.exp(yc.spot(1)) - 1
    a = Bond(principal=1000,
             maturity=1,
             coupon=coupon,
             freq=1,
             yc=yc)
    print(a.price)
    print(a.duration())
    assert a.duration() == 1
    print(a.modified_duration())

    coupon = math.exp(yc.spot(3)) - 1
    b = Bond(principal=1000,
             maturity=3,
             coupon=coupon,
             freq=3,
             yc=yc)
    print(b.price)
    print(b.duration())
    assert b.duration() == 3
    print(b.modified_duration())

    coupon = 0.0207813716493259
    b = Bond(principal=1000,
             maturity=3,
             coupon=coupon,
             freq=1,
             yc=yc)
    print(b.price)
    # assert int(b.price) == 1000
    print(b.duration())
    print(b.modified_duration())


if __name__ == "__main__":
    test_bond()
