from qe_macro_calc.entities.entity import Entity

class Bank(Entity):
    def __init__(self, name, initial_capital, max_bid, initial_interest_rate_markup, can_sell_bonds, risk_appetite, strategy):
        super().__init__(name, initial_capital, max_bid)
        self.interest_rate_markup = initial_interest_rate_markup
        self.loan_history = []
        self.bonds_owned = 0
        self.cash_reserves = initial_capital
        self.can_sell_bonds = can_sell_bonds
        self.risk_appetite = risk_appetite
        self.strategy = strategy

    def lend_to_consumers(self, consumers, good_price, margin_requirement, base_interest_rate):
        self.strategy.lend_to_consumers(self, consumers, good_price, margin_requirement, base_interest_rate)