from selenium.webdriver.common.by import By
from selenium import webdriver
import numpy as np


class Rates(object):
    path = r'C:\Users\kmato\PycharmProjects\chromedriver_win32\chromedriver.exe'
    url = \
        'https://www.ecb.europa.eu/stats/financial_markets_and_interest_rates/euro_area_yield_curves/html/index.en.html'

    def get(self):
        # Start the WebDriver and load the page
        driver = webdriver.Chrome(self.path)
        driver.get(self.url)

        # Get the table
        driver.execute_script('javascript:charts[0].switchDimension(1,1);')
        table = driver.find_element_by_class_name('ecb-contentTable')

        # Get the data and store it in a list/array
        result = []
        rows = table.find_elements(By.TAG_NAME, "tr")
        for i, row in enumerate(rows[1:]):
            data = row.find_elements(By.TAG_NAME, "td")[1]
            result.append(float(data.text))
        return np.array(result)
