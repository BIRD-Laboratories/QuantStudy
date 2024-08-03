class Entity:
    def __init__(self, id, initial_capital):
        self.id = id
        self.initial_capital = initial_capital
        self.income = 0
        self.outcome = 0
        self.total_money = initial_capital

    def update_money(self, income, outcome):
        self.income += income
        self.outcome += outcome
        self.total_money += (income - outcome)

class Bank(Entity):
    def __init__(self, id, initial_capital, max_bid, initial_interest_rate_markup, can_sell_bonds, risk_appetite, strategy):
        super().__init__(id, initial_capital)
        self.max_bid = max_bid
        self.initial_interest_rate_markup = initial_interest_rate_markup
        self.can_sell_bonds = can_sell_bonds
        self.risk_appetite = risk_appetite
        self.strategy = strategy
        self.bonds_owned = 0

class Consumer(Entity):
    def __init__(self, id, initial_capital, max_bid, age, salary, fixed_expenses, risk_appetite, strategy):
        super().__init__(id, initial_capital)
        self.max_bid = max_bid
        self.age = age
        self.salary = salary
        self.fixed_expenses = fixed_expenses
        self.risk_appetite = risk_appetite
        self.strategy = strategy

class Goods:
    def __init__(self, categories, weights, initial_price_level, parameters):
        self.categories = categories
        self.weights = weights
        self.initial_price_level = initial_price_level
        self.parameters = parameters
        self.prices = {cat: weight * initial_price_level for cat, weight in zip(categories, weights)}

class Company(Entity):
    def __init__(self, id, initial_capital, max_bid):
        super().__init__(id, initial_capital)
        self.max_bid = max_bid
    def __init__(self, name, initial_capital, max_bid):
        self.name = name
        self.capital = initial_capital
        self.max_bid = max_bid
