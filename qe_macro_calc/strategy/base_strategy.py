import jax
import jax.numpy as jnp
import jax.random as jrandom
import logging

class BaseStrategy:
    def __init__(self, categories):
        self.key = jrandom.PRNGKey(0)  # Initialize the key with a seed value
        self.categories = categories  # Add categories attribute
        self.bonds_purchased = 0  # Track the number of bonds purchased during quantitative easing

    def adjust_interest_rate(self, inflation, gdp_growth, unemployment_rate, interest_rate, interest_rate_adjustment, recession_status):
        # Lower interest rate until it reaches 0.025 or inflation is too high
        while interest_rate > 0.025 and inflation <= 0.03:
            interest_rate -= 0.01
            logging.info(f"Interest rate adjusted to: {interest_rate}")

        # Ensure interest rate does not go below 0.025
        interest_rate = max(interest_rate, 0.025)

        return jnp.float32(interest_rate)

    def implement_quantitative_easing(self, bond_price, recession_status, banks):
        # Custom logic for implementing quantitative easing
        if recession_status:
            new_bond_price = self.buy_bonds_from_banks(bond_price, banks)
            self.bonds_purchased += len(banks)  # Increment the count of bonds purchased
        else:
            # Create bonds out of thin air during non-recession times
            bonds_created = len(banks)  # Example: create one bond per bank
            self.bonds_purchased += bonds_created
            logging.info(f"Created {bonds_created} bonds out of thin air.")
            new_bond_price = bond_price * 1.02
            if self.bonds_purchased > 0:
                self.sell_bonds_to_banks(bond_price, banks)

        return jnp.float32(new_bond_price)

    def buy_bonds_from_banks(self, bond_price, banks):
        # Logic for buying bonds from banks during quantitative easing
        bonds_bought = 0
        for bank in banks:
            if bank.bonds_owned > 0:
                if bank.sell_bonds(bond_price):
                    bonds_bought += 1
        self.bonds_purchased += bonds_bought
        return jnp.float32(bond_price * 0.95)

    def sell_bonds_to_banks(self, bond_price, banks):
        # Logic for selling bonds back to banks after a recession
        bonds_sold = 0
        for bank in banks:
            if self.bonds_purchased > 0:
                if bank.invest_in_bonds(bond_price, 0.02):  # Assuming a bond yield of 2%
                    self.bonds_purchased -= 1
                    bonds_sold += 1
                    logging.info(f"Sold bond back to bank at price: {bond_price}")
        return bonds_sold

    def calculate_purchase_probabilities(self, consumer, good_prices, interest_rate):
        # Custom logic for calculating purchase probabilities
        return [price * (1 + interest_rate) for price in good_prices]

    def lend_to_consumers(self, bank, consumers, good_price, base_interest_rate):
        # Delegate lending to consumers to the Bank class
        bank.lend_to_consumers(consumers, good_price, base_interest_rate)
