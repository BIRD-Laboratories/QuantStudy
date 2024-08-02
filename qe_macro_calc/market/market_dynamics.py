import jax.numpy as jnp
import jax.random as jrandom
import logging

class MarketDynamics:
    def __init__(self, goods, consumers, interest_rate, recession_status, strategy, key):
        self.goods = goods
        self.consumers = consumers
        self.interest_rate = interest_rate
        self.recession_status = recession_status
        self.key = key
        self.strategy = strategy  
        logging.info("MarketDynamics instance created")

    def update_market_dynamics(self):
        logging.info("Updating market dynamics")
        self.goods.update_prices(self.interest_rate, self.recession_status)
        inflation = self.goods.calculate_inflation()

        good_prices = jnp.array(list(self.goods.prices.values()))
        purchase_probs = jnp.array([consumer.calculate_purchase_probabilities(good_prices, self.interest_rate) for consumer in self.consumers])

        buyers = jrandom.bernoulli(self.key, purchase_probs, (len(self.consumers), len(self.goods.categories)))
        sellers = jrandom.bernoulli(self.key, 0.1, (len(self.consumers), len(self.goods.categories)))

        bid_ask_spread = jnp.sum(buyers, axis=0) - jnp.sum(sellers, axis=0)
        price_adjustments = 1 + 0.01 * bid_ask_spread / len(self.consumers)
        new_prices = good_prices * price_adjustments

        for i, category in enumerate(self.goods.categories):
            self.goods.prices[category] = new_prices[i]

        return new_prices, buyers, sellers, inflation