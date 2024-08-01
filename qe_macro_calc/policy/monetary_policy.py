from qe_macro_calc.strategy.base_strategy import BaseStrategy

class MonetaryPolicy:
    def __init__(self, params, strategy: BaseStrategy):
        self.params = params
        self.strategy = strategy
        self.capital = 0
        self.max_bid = 0
        self.fixed_interest_rate = None
        self.balance_sheet = 0
        self.market_share_limit = params['market_share_limit_fed']
        self.bonds_owned = 0

    def adjust_interest_rate(self, inflation, gdp_growth, unemployment_rate, interest_rate, interest_rate_adjustment, recession_status):
        return self.strategy.adjust_interest_rate(inflation, gdp_growth, unemployment_rate, interest_rate, interest_rate_adjustment, recession_status)

    def implement_quantitative_easing(self, bond_price, recession_status):
        return self.strategy.implement_quantitative_easing(bond_price, recession_status)
