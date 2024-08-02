import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.lax as lax
from jax import pmap
from ..entities.goods import Goods
from ..entities.consumer import Consumer
from ..entities.bank import Bank
from ..entities.company import Company
from ..market.market import Market
from ..market.market_dynamics import MarketDynamics
from ..metrics.gdp import GDP
from ..strategy.base_strategy import BaseStrategy

class Simulation:
    def __init__(self, params, key):
        self.params = params
        self.key = key
        self.strategy = BaseStrategy(self.params['goods_categories'])  # Initialize with categories
        self.initialize_entities()

    def initialize_entities(self):
        self.ages, self.salaries, self.fixed_expenses, self.wage_increases, _ = self.initialize_population(self.params['size'], self.params['initial_price_level'], self.params['initial_price_level'] * 0.1, self.key)
        self.interest_rate = self.params['initial_interest_rate']
        self.good_price_history = jnp.zeros((self.params['num_rounds'], len(self.params['goods_categories'])))
        self.interest_rate_history = jnp.zeros(self.params['num_rounds'])
        self.interest_rate_history = self.interest_rate_history.at[0].set(self.params['initial_interest_rate'])
        self.transaction_history =  jnp.zeros((3))  # Initialize with fixed length
        self.inflation_history = jnp.zeros(self.params['num_rounds'])  # Initialize as a JAX array
        self.composite_inflation_history = jnp.zeros(self.params['num_rounds'])  # Initialize as a JAX array
        self.real_gdp_history = jnp.zeros(self.params['num_rounds'])  # Initialize as a JAX array

        self.banks = [
            Bank(f"Bank {i}", 1e9, 2 * self.params['initial_price_level'], jrandom.uniform(self.key, (), minval=0.005, maxval=0.02), self.params['banks_can_sell_bonds'][i], jrandom.uniform(self.key, (), minval=0.1, maxval=0.9), self.strategy)
            for i in range(self.params['num_banks'])
        ]
        self.consumers = [
            Consumer(self.salaries[i], 2 * self.params['initial_price_level'], self.ages[i], self.salaries[i], self.fixed_expenses[i], jrandom.uniform(self.key, (), minval=0.3, maxval=0.9), self.strategy)
            for i in range(self.params['size'])
        ]

        self.recession_rounds = jrandom.choice(self.key, jnp.arange(1, self.params['num_rounds'] - self.params['recession_duration_max']), (self.params['recessions'],), replace=False)
        self.recession_durations = jrandom.randint(self.key, (self.params['recessions'],), self.params['recession_duration_min'], self.params['recession_duration_max'] + 1)

        self.fed_bond_history = jnp.zeros(self.params['num_rounds'])
        self.bank_bond_history = jnp.zeros((self.params['num_rounds'], self.params['num_banks']))
        self.bank_cash_history = jnp.zeros((self.params['num_rounds'], self.params['num_banks']))

        self.goods = Goods(self.params['goods_categories'], self.params['goods_weights'], self.params['initial_price_level'], self.params)
        self.market = Market(liquidity_factor=0.05)
        self.gdp = GDP(self.params['goods_categories'])

        # Initialize Company instances if needed
        self.companies = [
            Company(f"Company {i}", 1e6, 1e5)  # Use positional arguments as per the Company class definition
            for i in range(self.params['num_companies'])
        ]

    def initialize_population(self, size, initial_good_price, good_price_std_dev, key):
        ages = jrandom.randint(key, (size,), 0, 61)
        salaries = jrandom.uniform(key, (size,), minval=30000, maxval=200000)
        median_salary = jnp.median(salaries)
        fixed_expenses = median_salary * jrandom.uniform(key, (size,), minval=0.2, maxval=0.5)
        fixed_expenses *= (salaries / median_salary) ** 0.5
        wage_increases = jrandom.uniform(key, (size,), minval=0.02, maxval=0.05) / 8
        standard_normal_prices = jrandom.normal(key, (size,))

        good_prices = initial_good_price + standard_normal_prices * good_price_std_dev
        good_prices = jnp.clip(good_prices, 0, None)
        return ages, salaries, fixed_expenses, wage_increases, good_prices

    def run_round(self, round, state):
        recession_status = jax.lax.cond(
            jnp.any(round == self.recession_rounds),
            lambda _: True,
            lambda _: False,
            operand=None
        )

        self.key, subkey = jrandom.split(self.key)
        market_dynamics = MarketDynamics(self.goods, self.consumers, self.interest_rate, recession_status, self.strategy, subkey)
        good_prices, buyers, sellers, inflation = market_dynamics.update_market_dynamics()
        state[0] = self.update_good_price_history(round, good_prices, state[0])
        state[1] = self.update_transaction_history(round, self.consumers, self.params['goods_categories'], state[1], good_prices, buyers, sellers)

        self.interest_rate = self.strategy.adjust_interest_rate(inflation, 0, 0, self.interest_rate, self.params['interest_rate_adjustment'], recession_status)
        state[2] = state[2].at[round].set(self.interest_rate)
        state[3] = state[3].at[round].set(self.strategy.implement_quantitative_easing(self.params['initial_bond_price'], recession_status, self.banks))

        for bank in self.banks:
            bank.lend_to_consumers(self.consumers, good_prices[0], self.interest_rate)
            bank.invest_in_bonds(self.params['initial_bond_price'], self.params['bond_yield'])
            state[4] = state[4].at[round, self.banks.index(bank)].set(bank.bonds_owned)
            state[5] = state[5].at[round, self.banks.index(bank)].set(bank.cash_reserves)

        # Update good price history with adjusted prices
        state[0] = self.update_good_price_history(round, good_prices, state[0])
        state[6] = state[6].at[round].set(inflation)
        gdp = self.gdp.calculate_gdp(good_prices, buyers)
        return state

    def update_transaction_history(self, round, consumers, goods_categories, transaction_history, good_prices, buyers, sellers):
        buy_amounts = jnp.sum(buyers * good_prices, axis=1)
        sell_amounts = jnp.sum(sellers * good_prices, axis=1)

        Consumer.update_capitals(consumers, buy_amounts, sell_amounts)

        total_buy_amount = jnp.sum(buy_amounts)
        total_sell_amount = jnp.sum(sell_amounts)

        transaction_history = jnp.array([round, total_buy_amount, total_sell_amount])
        return transaction_history

    def update_good_price_history(self, round, good_prices, good_price_history):
        for i, price in enumerate(good_prices):
            good_price_history = good_price_history.at[round, i].set(price)
        return good_price_history

    def run(self):
        state = [
            self.good_price_history,
            self.transaction_history,
            self.interest_rate_history,
            self.fed_bond_history,
            self.bank_bond_history,
            self.bank_cash_history,
            self.inflation_history
        ]

        for round in range(1, self.params['num_rounds']):
            state = self.run_round(round, state)

        print(f"Final Interest Rate: {state[2][self.params['num_rounds'] - 1]:.4f}")
        print(f"Fed bond amount: {state[3][self.params['num_rounds'] - 1]}")
        print(f"Final Good Prices: {state[0][self.params['num_rounds'] - 1]}")
        return state

# Example usage of pmap for parallel execution
def parallel_simulations(params_list, key):
    keys = jrandom.split(key, len(params_list))
    simulations = [Simulation(params, k) for params, k in zip(params_list, keys)]
    results = pmap(lambda sim: sim.run())(simulations)
    return results