import jax
import jax.numpy as jnp
import jax.random as jrandom
import logging

class BaseStrategy:
    def __init__(self, categories):
        self.key = jrandom.PRNGKey(0)  # Initialize the key with a seed value
        self.categories = categories  # Add categories attribute

    def adjust_interest_rate(self, inflation, gdp_growth, unemployment_rate, interest_rate, interest_rate_adjustment, recession_status):
        # Adjust interest rate based on economic indicators using jax.lax.cond
        interest_rate = jax.lax.cond(
            inflation > 0.03,  # If inflation is above 3%
            lambda rate: rate + 0.01,
            lambda rate: rate,
            interest_rate
        )

        interest_rate = jax.lax.cond(
            inflation < 0.01,  # If inflation is below 1%
            lambda rate: rate - 0.01,
            lambda rate: rate,
            interest_rate
        )

        return jnp.float32(interest_rate)

    def implement_quantitative_easing(self, bond_price, recession_status, banks):
        # Custom logic for implementing quantitative easing using jax.lax.cond
        new_bond_price = jax.lax.cond(
            recession_status,
            lambda _: self.buy_bonds_from_banks(bond_price, banks),
            lambda _: bond_price * 1.02,
            operand=None
        )

        return jnp.float32(new_bond_price)

    def buy_bonds_from_banks(self, bond_price, banks):
        # Logic for buying bonds from banks during quantitative easing
        for bank in banks:
            if bank.bonds_owned > 0:
                bank.bonds_owned -= 1
                bank.cash_reserves += bond_price
        return jnp.float32(bond_price * 0.95)

    def calculate_purchase_probabilities(self, consumer, good_prices, interest_rate):
        # Custom logic for calculating purchase probabilities
        return [price * (1 + interest_rate) for price in good_prices]

    def lend_to_consumers(self, bank, consumers, good_price, margin_requirement, base_interest_rate):
        # Custom logic for lending to consumers
        for consumer in consumers:
            # Remove credit score check and directly adjust capital
            consumer.capital += good_price * 0.5