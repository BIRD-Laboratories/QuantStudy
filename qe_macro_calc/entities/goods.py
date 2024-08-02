import jax.numpy as jnp
import jax.random as jrandom
import jax
import logging

class Goods:
    def __init__(self, categories, weights, initial_price_level, parameters):
        if len(categories) != len(weights):
            raise ValueError("Categories and weights must have the same length.")

        self.categories = categories
        self.weights = {cat: weight for cat, weight in zip(categories, weights)}
        self.prices = {cat: weight * initial_price_level for cat, weight in zip(categories, weights)}
        self.parameters = parameters
        logging.info("Goods instance created")

    def update_prices(self, interest_rate, recession_status):
        logging.info(f"Updating prices with recession_status: {recession_status}")
        for category in self.categories:
            old_price = self.prices[category]
            self.prices[category] = jax.lax.cond(
                recession_status,
                lambda price: price * (1 - self.parameters['recession_severity']),
                lambda price: price * (1 + jrandom.uniform(jax.random.PRNGKey(0), (), minval=-0.01, maxval=0.03)),
                self.prices[category]
            )
            new_price = self.prices[category]
            logging.info(f"Category: {category}, Old Price: {old_price}, New Price: {new_price}")

    def calculate_inflation(self):
        total_weight = sum(self.weights.values())
        weighted_prices = [self.prices[cat] * self.weights[cat] for cat in self.categories]
        return sum(weighted_prices) / total_weight

    def get_price(self, category):
        return self.prices.get(category, None)

    def get_all_prices(self):
        return self.prices