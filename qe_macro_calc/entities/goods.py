import jax.random as jrandom
import jax

class Goods:
    def __init__(self, categories, weights, initial_price_level):
        if len(categories) != len(weights):
            raise ValueError("Categories and weights must have the same length.")

        self.categories = categories
        self.weights = {cat: weight for cat, weight in zip(categories, weights)}
        self.prices = {cat: weight * initial_price_level for cat, weight in zip(categories, weights)}

    def update_prices(self, interest_rate, recession_status):
        bond_rate = 0.007 + interest_rate
        for category in self.categories:
            if recession_status == "in_progress":
                self.prices[category] *= (1 - PARAMETERS['recession_severity'])
            elif category == 'Housing':
                self.prices[category] *= (1 + bond_rate)
            else:
                self.prices[category] *= (1 + jrandom.uniform(jax.random.PRNGKey(0), (), minval=-0.01, maxval=0.03))

    def calculate_inflation(self):
        total_weight = sum(self.weights.values())
        weighted_prices = [self.prices[cat] * self.weights[cat] for cat in self.categories]
        return sum(weighted_prices) / total_weight

    def get_price(self, category):
        return self.prices.get(category, None)

    def get_all_prices(self):
        return self.prices