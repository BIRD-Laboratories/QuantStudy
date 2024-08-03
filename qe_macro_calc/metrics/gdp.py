import numpy as np

class GDP:
    def __init__(self, goods_categories):
        self.goods_categories = goods_categories
        self.gdp_history = []

    def calculate_gdp(self, good_prices, buyers):
        # Repeat good_prices for each consumer
        repeated_good_prices = np.tile(good_prices, (len(buyers), 1))
        gdp = np.sum(buyers[:, np.newaxis] * repeated_good_prices)
        self.gdp_history.append(gdp)
        return gdp

    def get_gdp_history(self):
        return self.gdp_history