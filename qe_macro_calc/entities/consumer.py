from qe_macro_calc.entities.entity import Entity

class Consumer(Entity):
    def __init__(self, initial_capital, max_bid, age, salary, fixed_expenses, credit_score, strategy_module):
        super().__init__("", initial_capital, max_bid)
        self.age = age
        self.salary = salary
        self.fixed_expenses = fixed_expenses
        self.credit_score = credit_score
        self.strategy = strategy_module  # Ensure this is the module, not a string path

    def calculate_purchase_probabilities(self, good_prices, interest_rate):
        return self.strategy.calculate_purchase_probabilities(self, good_prices, interest_rate)