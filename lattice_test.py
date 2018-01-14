from lattice import CRR, Convertible, Oracle


def test_crr():
    apple = CRR(25, "AAPL", "USD", "call", 1, 170)
    print(apple.option.maturity)
    print(apple.option.price)
    print(apple.tree[0][0])


def test_convertible():
    apple = Convertible(steps=8,
                        principal=1000,
                        maturity=2,
                        coupon=.02,
                        coupon_freq=1,
                        spread=0.01,
                        ticker="AAPL",
                        ccy="USD",
                        con_ratio=5,
                        call=1100)
    print(apple.tree[0])

    pear = Oracle(steps=8,
                  principal=1000,
                  maturity=2,
                  coupon=.02,
                  coupon_freq=1,
                  spread=0.01,
                  ticker="AAPL",
                  ccy="USD",
                  con_ratio=5,
                  call=1100)
    print(pear.tree[0])


if __name__ == "__main__":
    test_convertible()
