from ecb import Scraper


def test():
    yield_curve = Scraper().data()
    assert len(yield_curve) == 33
    print("Test function executed successfully \n",
          yield_curve)


if __name__ == "__main__":
    test()
