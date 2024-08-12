import numpy as np

class Kernels:
    def __init__(self):
        pass

    def adjust_interest_rate(self, interest_rate, inflation, interest_rate_adjustment):
        if inflation > 0:
            interest_rate_adjustment = inflation * 0.0275
        else:
            interest_rate_adjustment = inflation * 0.0175
        return interest_rate + interest_rate_adjustment

    def adjust_prices(self, goods, interest_rate, num_consumers):
        new_prices = np.empty_like(goods)
        for i in range(len(goods)):
            purchase_prob = goods[i] * (1 + interest_rate)
            buy = np.where(purchase_prob > 0.5, 1, 0)
            sell = np.where(purchase_prob < 0.1, 1, 0)
            bid_ask_spread = buy - sell
            price_adjustment = 1 + 0.01 * bid_ask_spread / num_consumers
            new_prices[i] = goods[i] * price_adjustment
        return new_prices

    def calculate_inflation(self, round, num_rounds=128):
        # Simulate inflation as a sine wave with random fluctuations
        base_inflation = np.sin(2 * np.pi * 128 / num_rounds)
        fluctuation = np.random.normal(0, 0.1)  # Random noise with mean 0 and standard deviation 0.1
        inflation = base_inflation + fluctuation
        return inflation

    def update_money_supply_and_salary(self, money_supply, salary, inflation, interest_rate, bond_price, money_supply_increment):
        new_money_supply = money_supply * (1 + (1 / (interest_rate * 2))) if interest_rate < 0.05 else 1 - interest_rate * 5
        updated_money_supply = money_supply + money_supply_increment
        updated_salary = salary * (1 + inflation)
        return updated_money_supply, updated_salary

    def fused_kernel(self, banks, consumers, goods, interest_rate, buy_amounts, sell_amounts, gdp_growth, unemployment_rate, interest_rate_adjustment, bond_price, money_supply, salary, money_supply_increment, last_prices, num_round):
        num_consumers = len(consumers)
        num_goods = len(goods)

        new_prices = self.adjust_prices(goods, interest_rate, num_consumers)
        inflation = self.calculate_inflation(round)
        interest_rate = self.adjust_interest_rate(interest_rate, inflation, interest_rate_adjustment)
        updated_money_supply, updated_salary = self.update_money_supply_and_salary(money_supply, salary, inflation, interest_rate, bond_price, money_supply_increment)

        return banks, consumers, goods, interest_rate, float(inflation), 0, 0, updated_money_supply, updated_salary