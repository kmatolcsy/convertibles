from lattice3 import GS
from simulation3 import LSM
from bsmarket3 import Bond
from matplotlib import style
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


style.use("ggplot")


tesla_lsm = LSM(iterations=2500, order=2, default=0, month=12, symbol="TSLA")
print(tesla_lsm.get_price())
quit()
# tesla_gs = GS(spread=0, month=1, symbol="TSLA")
# print(tesla_gs.get_price())
# quit()


def pricing_function(parameter, month, symbol):
    try:
        model = LSM(iterations=2500, order=2, default=parameter, month=month, symbol=symbol)
        price = model.get_price()
        return price
    except Exception as e:
        print(parameter, e)
        return 0


def calibration(market_price, month, symbol, epsilon=0.001):
    lo = 0.25
    hi = 0.45
    counter = 0
    while counter < 5000:
        counter += 1
        mid = (hi + lo) / 2
        if abs(pricing_function(mid, month, symbol) - market_price) < epsilon:
            return mid
        if np.sign(pricing_function(mid, month, symbol) - market_price) == \
                np.sign(pricing_function(lo, month, symbol) - market_price):
            lo = mid
        else:
            hi = mid
        if round(mid, 6) == round((hi + lo) / 2, 6):
            return mid, "*"


def main():
    symbol = "TSLA"
    df = pd.read_excel("data/CONVERTIBLE.xls", index_col=0, header=0)
    df = df.transpose()
    for month in range(12, 9, -1):
        bond_price = df[month][symbol] * 10

        # visualization of the calibration
        # x = [i/1000 for i in range(150, 700, 25)]
        # y = [pricing_function(xi, month, symbol) for xi in x]
        # z = [bond_price for xi in x]
        #
        # plt.plot(x, y, 'o', x, z, '-')
        # plt.show()

        print(calibration(bond_price, month, symbol))
        return None


if __name__ == "__main__":
    main()
