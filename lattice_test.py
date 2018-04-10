from lattice import CRR, Convertible, Oracle
from bsmarket import YieldCurve
import numpy as np


def test_crr():
    apple = CRR(25, "AAPL", "USD", "call", 1, 170)
    print(apple.option.maturity)
    print(apple.option.price)
    print(apple.tree[0][0])


def test_convertible():
    apple = Convertible(steps=8,
                        principal=1000,
                        maturity=2,
                        coupon=.0,
                        coupon_freq=1,
                        spread=0.0,
                        ticker="AAPL",
                        ccy="USD",
                        con_ratio=5,
                        call=1100)
    print(apple.tree[0])

    pear = Oracle(steps=8,
                  principal=1000,
                  maturity=2,
                  coupon=.0,
                  coupon_freq=1,
                  spread=0.0,
                  ticker="AAPL",
                  ccy="USD",
                  con_ratio=5,
                  call=1100)
    print(pear.tree[0])


def test_consistency():
    yc = YieldCurve("USD")
    coupon = np.exp(yc.spot(1)) - 1

    bond = Convertible(steps=6,
                       principal=1000,
                       maturity=1,
                       coupon=coupon,
                       coupon_freq=1,
                       spread=0,
                       ticker="AAPL",
                       ccy="USD",
                       con_ratio=0.00001,
                       call=99999)
    print(bond.tree[0])
    assert bond.tree[0][0][1] == 1000

    conv = Convertible(steps=6,
                       principal=1000,
                       maturity=1,
                       coupon=coupon,
                       coupon_freq=1,
                       spread=0,
                       ticker="AAPL",
                       ccy="USD",
                       con_ratio=5,
                       call=99999)
    print(conv.tree[0])

    option = CRR(steps=6,
                 ticker="AAPL",
                 ccy="USD",
                 warrant=True,
                 strike=200)
    print(option.tree[0])
    print(option.tree[0][0][1] * 5 + 1000)
    assert round(conv.tree[0][0][1], 8) == round(option.tree[0][0][1] * 5 + 1000, 8)


if __name__ == "__main__":
    test_consistency()
