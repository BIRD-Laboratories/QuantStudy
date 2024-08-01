import jax.numpy as jnp

class GDP:
    def __init__(self, goods_categories):
        self.goods_categories = goods_categories
        self.gdp_history = []

    def calculate_gdp(self, good_prices, buyers):
        gdp = jnp.sum(buyers * good_prices)
        self.gdp_history.append(gdp)
        return gdp

    def get_gdp_history(self):
        return self.gdp_history