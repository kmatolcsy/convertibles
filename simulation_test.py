from simulation import LSM, Example
import numpy as np


def test_example():
    stock = 36
    strike = 40
    while True:
        print(stock, strike)
        ex = Example(stock=stock,
                     strike=strike,
                     maturity=1.,
                     rate=.06,
                     sigma=.2)
        print("put-call parity")
        print(str(ex.bs_call()) + ' - ' + str(ex.bs_put()) + ' = ' + str(ex.forward()))
        assert round(ex.bs_call() - ex.bs_put(), 6) == round(ex.forward(), 6)
        print("convergence")
        print(str(ex.eu_call) + ' --> ' + str(ex.bs_call()))
        print(str(ex.eu_put) + ' --> ' + str(ex.bs_put()))
        print("american mc vs. european bs")
        print(str(ex.am_call) + ' ~ ' + str(ex.bs_call()))
        print(str(ex.am_put) + ' > ' + str(ex.bs_put()))
        assert ex.am_put > ex.bs_put()
        print('test function end')
        print('\n')
        stock *= 2
        strike *= 2


def test_lsm():
    np.random.seed(123)
    ex = LSM(steps=8,
             its=2500,
             order=2,
             principal=1000,
             maturity=1,
             coupon=.01,
             coupon_freq=0.5,
             ticker="AAPL",
             ccy="USD",
             con_ratio=5,
             call=99999)


if __name__ == "__main__":
    # test_example()
    test_lsm()
