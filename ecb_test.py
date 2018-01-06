from ecb import Rates


def test():
    rates = Rates().get()
    assert len(rates) == 33
    print("Test function executed successfully \n",
          rates)


if __name__ == "__main__":
    test()
