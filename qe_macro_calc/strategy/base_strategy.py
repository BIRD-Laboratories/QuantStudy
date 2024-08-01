class BaseStrategy:
    def adjust_interest_rate(self, inflation, gdp_growth, unemployment_rate, interest_rate, interest_rate_adjustment, recession_status):
        # Custom logic for adjusting interest rate
        if recession_status == 'in_progress':
            return interest_rate - 0.01
        return interest_rate + 0.005

    def implement_quantitative_easing(self, bond_price, recession_status):
        # Custom logic for implementing quantitative easing
        if recession_status == 'in_progress':
            return bond_price * 0.95
        return bond_price * 1.02

    def calculate_purchase_probabilities(self, consumer, good_prices, interest_rate):
        # Custom logic for calculating purchase probabilities
        return [price * (1 + interest_rate) for price in good_prices]

    def lend_to_consumers(self, bank, consumers, good_price, margin_requirement, base_interest_rate):
        # Custom logic for lending to consumers
        for consumer in consumers:
            if consumer.credit_score > 0.5:
                consumer.capital += good_price * 0.5