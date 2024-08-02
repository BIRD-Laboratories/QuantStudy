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

    def lend_to_consumers(self, consumers, good_price, base_interest_rate):
        # Calculate margin requirement based on interest rate
        margin_requirement = max(0.05 - (base_interest_rate * 0.1), 0.01)  # Example formula

        for consumer in consumers:
            if consumer.capital >= good_price * margin_requirement:
                consumer.capital += good_price * (1 - margin_requirement)
                self.cash_reserves -= good_price * (1 - margin_requirement)
                self.loan_history.append((consumer, good_price, base_interest_rate + self.interest_rate_markup))

    def sell_bonds(self, bond_price):
        if self.can_sell_bonds and self.bonds_owned > 0:
            self.bonds_owned -= 1
            self.cash_reserves += bond_price

    def invest_in_bonds(self, bond_price, bond_yield):
        # Example logic for investing in bonds
        if self.cash_reserves >= bond_price:
            self.bonds_owned += 1
            self.cash_reserves -= bond_price
            self.cash_reserves += bond_price * bond_yield