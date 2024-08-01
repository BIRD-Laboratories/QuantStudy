import jax.numpy as jnp
import jax.random as jrandom

class MarketDynamics:
    def __init__(self, goods, consumers, interest_rate, recession_status, strategy, key):
        self.goods = goods
        self.consumers = consumers
        self.interest_rate = interest_rate
        self.recession_status = recession_status
        self.key = key
        self.strategy = strategy  

    def update_market_dynamics(self):
        self.goods.update_prices(self.interest_rate, self.recession_status)
        inflation = self.goods.calculate_inflation()

        good_prices = jnp.array(list(self.goods.prices.values()))
        purchase_probs = jnp.array([consumer.calculate_purchase_probabilities(good_prices, self.interest_rate) for consumer in self.consumers])

        buyers = jrandom.bernoulli(self.key, purchase_probs, (len(self.consumers), len(self.goods.categories)))
        sellers = jrandom.bernoulli(self.key, 0.1, (len(self.consumers), len(self.goods.categories)))

        bid_ask_spread = jnp.sum(buyers, axis=0) - jnp.sum(sellers, axis=0)
        for i, category in enumerate(self.goods.categories):
            self.goods.prices[category] *= (1 + 0.01 * bid_ask_spread[i] / len(self.consumers))

        return list(self.goods.prices.values()), buyers, sellers, inflation
