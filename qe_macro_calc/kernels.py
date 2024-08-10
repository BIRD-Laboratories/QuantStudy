import numpy as np

class Kernels:
    def __init__(self):
        pass

    def fused_kernel(self, banks, consumers, goods, interest_rate, buy_amounts, sell_amounts, gdp_growth, unemployment_rate, interest_rate_adjustment, bond_price, money_supply, salary, money_supply_increment, last_prices):
        num_banks = len(banks)
        num_consumers = len(consumers)
        num_goods = len(goods)

        new_prices = np.empty_like(goods)
        inflation = np.empty(1, dtype=np.float32)
        bank_bond_buying = np.empty(1, dtype=np.float32)
        bank_bond_selling = np.empty(1, dtype=np.float32)
        updated_money_supply = np.empty(1, dtype=np.float32)
        updated_salary = np.empty(1, dtype=np.float32)

        # Ensure last_prices is contiguous
        last_prices = np.ascontiguousarray(last_prices, dtype=np.float32)

        for gid in range(num_consumers):
            for i in range(num_goods):
                purchase_prob = goods[i] * (1 + interest_rate)
                buy = np.where(purchase_prob > 0.5, 1, 0)
                sell = np.where(purchase_prob < 0.1, 1, 0)
                bid_ask_spread = buy - sell
                price_adjustment = 1 + 0.01 * bid_ask_spread / num_consumers
                new_prices[i] = goods[i] * price_adjustment

            # Ensure the index is within bounds
            if gid * 4 + 3 < len(consumers):
                consumers[gid * 4 + 3] -= buy_amounts[gid]
                consumers[gid * 4 + 3] += sell_amounts[gid]
            else:
                raise IndexError(f"Index out of bounds: gid * 4 + 3 = {gid * 4 + 3}, len(consumers) = {len(consumers)}")

        for gid in range(num_consumers, num_consumers + num_banks):
            bank_id = gid - num_consumers
            margin_requirement = 0.05 - (interest_rate * 0.1)
            margin_requirement = max(margin_requirement, 0.01)
            if consumers[gid - num_consumers] >= goods[0] * margin_requirement:
                consumers[gid - num_consumers] += goods[0] * (1 - margin_requirement)
                banks[bank_id * 4 + 3] -= goods[0] * (1 - margin_requirement)

        total_prices = np.sum(new_prices)
        total_last_prices = np.sum(last_prices)
        average_price = total_prices / num_goods
        average_last_price = total_last_prices / num_goods
        inflation = (average_price / average_last_price) - 1

        if inflation > 0:
            interest_rate_adjustment = inflation * 0.01
        else:
            interest_rate_adjustment = inflation * 0.005
        interest_rate += interest_rate_adjustment

        new_money_supply = money_supply * (1 + (1 / (interest_rate * 2))) if interest_rate < 0.05 else 1 - interest_rate * 5
        for i in range(num_goods):
            new_prices[i] = new_money_supply * goods[i]

        total_bond_buying = 0
        total_bond_selling = 0
        for i in range(num_banks):
            banks[i * 4 + 2] += 1
            banks[i * 4 + 3] -= bond_price
            total_bond_selling += 1

        bank_bond_buying = total_bond_buying
        bank_bond_selling = total_bond_selling

        updated_money_supply = money_supply + total_bond_buying * bond_price - total_bond_selling * bond_price + money_supply_increment
        updated_salary = salary * (1 + inflation)

        return banks, consumers, goods, interest_rate, inflation, bank_bond_buying, bank_bond_selling, updated_money_supply, updated_salary