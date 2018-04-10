from lattice3 import GS
from simulation3 import LSM
from bsmarket3 import Bond, Stock
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def pricing_function(model, month, symbol, ir_stress=None, eq_stress=None):
    if model == "LSM":
        df = pd.read_excel("data/CREDIT_LSM.xls", index_col=0, header=0)
        df = df.transpose()
        default_parameter = df[month][symbol]
        model = LSM(iterations=2500, order=2, default=default_parameter, month=month, symbol=symbol,
                    ir_stress=ir_stress, eq_stress=eq_stress)
        return model.get_price()
    elif model == "GS":
        df = pd.read_excel("data/CREDIT_GS.xls", index_col=0, header=0)
        df = df.transpose()
        default_parameter = df[month][symbol]
        model = GS(spread=default_parameter, month=month, symbol=symbol, ir_stress=ir_stress, eq_stress=eq_stress)
        return model.get_price()


def capital_requirement(model, symbol):
    # for all month
    for month in range(12, 9, -1):
        scr = []
        market_price = pricing_function(model, month, symbol)

        # interest rate
        stress_price_ir = min(pricing_function(model=model, month=month, symbol=symbol, ir_stress="dec"),
                              pricing_function(model=model, month=month, symbol=symbol, ir_stress="inc"))
        scr.append(market_price - stress_price_ir)

        # equity
        stress_price_eq = pricing_function(model=model, month=month, symbol=symbol, eq_stress="type1")
        scr.append(market_price - stress_price_eq)

        # spread
        vanilla_bond = Bond(month=month, symbol=symbol, rating=None)
        scr.append(vanilla_bond.charge)

        # equity for comparison
        stock = Stock(month=month, symbol=symbol, stress="type1")

        print(scr, sum(scr), sum(scr) / market_price, stock.charge_pct)
    return None


if __name__ == "__main__":
    capital_requirement("GS", "TSLA")
