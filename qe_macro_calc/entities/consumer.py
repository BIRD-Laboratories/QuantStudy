import jax.numpy as jnp

class Consumer:
    def __init__(self, initial_capital, max_bid, age, salary, fixed_expenses, credit_score, strategy_module):
        self.capital = initial_capital
        self.max_bid = max_bid
        self.age = age
        self.salary = salary
        self.fixed_expenses = fixed_expenses
        self.credit_score = credit_score
        self.strategy = strategy_module  # Ensure this is the module, not a string path

    @staticmethod
    def from_arrays(initial_capitals, max_bids, ages, salaries, fixed_expenses, credit_scores, strategy_module):
        consumers = []
        for i in range(len(initial_capitals)):
            consumers.append(Consumer(initial_capitals[i], max_bids[i], ages[i], salaries[i], fixed_expenses[i], credit_scores[i], strategy_module))
        return consumers

    def calculate_purchase_probabilities(self, good_prices, interest_rate):
        essential_goods = ['Food', 'Energy']
        essential_indices = [i for i, category in enumerate(self.strategy.categories) if category in essential_goods]
        
        # Calculate purchase probabilities based on essential goods and other factors
        purchase_probs = [price * (1 + interest_rate) for price in good_prices]
        for idx in essential_indices:
            purchase_probs[idx] *= 1.2  # Increase probability for essential goods
        
        return purchase_probs

    @staticmethod
    def update_capitals(consumers, buy_amounts, sell_amounts):
        capitals = jnp.array([c.capital for c in consumers])
        capitals -= buy_amounts
        capitals += sell_amounts
        for i, consumer in enumerate(consumers):
            consumer.capital = capitals[i]