from ctypes import string_at
import jax
import jax.numpy as jnp
import jax.random as jrandom
from ..entities.goods import Goods
from ..entities.consumer import Consumer
from ..entities.bank import Bank
from ..policy.monetary_policy import MonetaryPolicy
from ..market.market import Market
from ..market.market_dynamics import MarketDynamics
from ..metrics.gdp import GDP
from ..strategy.base_strategy import BaseStrategy
import importlib.util

class Simulation:
    def __init__(self, params, key):
        self.params = params
        self.key = key
        self.initialize_entities()
    
    def load_strategy(self, strategy):
        if isinstance(strategy, str):
            spec = importlib.util.spec_from_file_location("custom_strategy", strategy)
            strategy_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(strategy_module)
            return strategy_module
        elif isinstance(strategy, type(importlib.util.module_from_spec(None))):
            return strategy
        else:
            raise TypeError("Expected a file path (str) or a module object")
    
    def initialize_entities(self):
        self.ages, self.salaries, self.fixed_expenses, self.wage_increases, _ = self.initialize_population(self.params['size'], self.params['initial_price_level'], self.params['initial_price_level'] * 0.1, self.key)
        self.interest_rate = self.params['initial_interest_rate']
        self.good_price_history = jnp.zeros((self.params['num_rounds'], len(self.params['goods_categories'])))
        self.interest_rate_history = jnp.zeros(self.params['num_rounds'])
        self.interest_rate_history = self.interest_rate_history.at[0].set(self.params['initial_interest_rate'])
        self.transaction_history = []
        self.inflation_history = []
        self.composite_inflation_history = []
        self.real_gdp_history = []

        self.strategy = self.load_strategy(self.params.get('strategy'))
        strategy = self.strategy
        self.monetary_policy = MonetaryPolicy(self.params, self.strategy)

        self.banks = [
            Bank(f"Bank {i}", 1e9, 2 * self.params['initial_price_level'], jrandom.uniform(self.key, (), minval=0.005, maxval=0.02), self.params['banks_can_sell_bonds'][i], jrandom.uniform(self.key, (), minval=0.1, maxval=0.9), strategy)
            for i in range(self.params['num_banks'])
        ]
        self.consumers = [
            Consumer(self.salaries[i], 2 * self.params['initial_price_level'], self.ages[i], self.salaries[i], self.fixed_expenses[i], jrandom.uniform(self.key, (), minval=0.3, maxval=0.9), strategy)
            for i in range(self.params['size'])
        ]

        self.recession_rounds = jrandom.choice(self.key, jnp.arange(1, self.params['num_rounds'] - self.params['recession_duration_max']), (self.params['recessions'],), replace=False)
        self.recession_durations = jrandom.randint(self.key, (self.params['recessions'],), self.params['recession_duration_min'], self.params['recession_duration_max'] + 1)

        self.fed_bond_history = jnp.zeros(self.params['num_rounds'])
        self.bank_bond_history = jnp.zeros((self.params['num_rounds'], self.params['num_banks']))
        self.bank_cash_history = jnp.zeros((self.params['num_rounds'], self.params['num_banks']))

        self.goods = Goods(self.params['goods_categories'], self.params['goods_weights'], self.params['initial_price_level'])
        self.market = Market(liquidity_factor=0.05)
        self.gdp = GDP(self.params['goods_categories'])

    def initialize_population(self, size, initial_good_price, good_price_std_dev, key):
        ages = jrandom.randint(key, (size,), 0, 61)
        salaries = jrandom.uniform(key, (size,), minval=30000, maxval=200000)
        median_salary = jnp.median(salaries)
        fixed_expenses = median_salary * jrandom.uniform(key, (size,), minval=0.2, maxval=0.5)
        fixed_expenses *= (salaries / median_salary) ** 0.5
        wage_increases = jrandom.uniform(key, (size,), minval=0.02, maxval=0.05) / 8
        standard_normal_prices = jrandom.normal(key, (size,))

# Adjust the mean and standard deviation
        good_prices = initial_good_price + standard_normal_prices * good_price_std_dev
        good_prices = jnp.clip(good_prices, 0, None)
        return ages, salaries, fixed_expenses, wage_increases, good_prices

    def run(self):
        strategy = self.strategy
        for round in range(1, self.params['num_rounds']):
            if round in self.recession_rounds:
                recession_status = 'in_progress'
            else:
                recession_status = 'over'
            self.key, subkey = jrandom.split(self.key)
            market_dynamics = MarketDynamics(self.goods, self.consumers, self.interest_rate, recession_status, strategy, subkey)
            good_prices, buyers, sellers, inflation = market_dynamics.update_market_dynamics()
            self.good_price_history = self.update_good_price_history(round, good_prices, self.good_price_history)
            self.transaction_history = self.update_transaction_history(round, self.consumers, self.params['goods_categories'], self.transaction_history, good_prices, buyers, sellers)

            self.interest_rate = self.monetary_policy.adjust_interest_rate(inflation, 0, 0, self.interest_rate, self.params['interest_rate_adjustment'], recession_status)
            self.interest_rate_history = self.interest_rate_history.at[round].set(self.interest_rate)
            self.fed_bond_history = self.fed_bond_history.at[round].set(self.monetary_policy.implement_quantitative_easing(self.params['initial_bond_price'], recession_status))

            for bank in self.banks:
                if round % 8 == 0:
                    bank.sell_bonds(self.params['initial_bond_price'])
                bank.lend_to_consumers(self.consumers, good_prices[0], self.params['margin_requirement'], self.interest_rate)
                bank.invest_in_bonds(self.params['initial_bond_price'], self.params['bond_yield'])
                self.bank_bond_history = self.bank_bond_history.at[round, self.banks.index(bank)].set(bank.bonds_owned)
                self.bank_cash_history = self.bank_cash_history.at[round, self.banks.index(bank)].set(bank.cash_reserves)

            gdp = self.gdp.calculate_gdp(good_prices, buyers)
            self.inflation_history.append(inflation)
            composite_inflation = jnp.mean(self.inflation_history[-12:]) if len(self.inflation_history) >= 12 else jnp.mean(self.inflation_history)
            self.composite_inflation_history.append(composite_inflation)
            real_gdp_growth = (gdp / (1 + composite_inflation)) - self.gdp.get_gdp_history()[-2] if len(self.gdp.get_gdp_history()) > 1 else 0
            self.real_gdp_history.append(real_gdp_growth)

        print(f"Final Interest Rate: {self.interest_rate:.4f}")
        print(f"Fed bond amount: {self.fed_bond_history[round]}")
        print(f"Final Good Prices: {good_prices}")
        appreciation_rates = jnp.diff(self.good_price_history, axis=0) / self.good_price_history[:-1]
        average_appreciation_rate = jnp.mean(appreciation_rates)
        target_rate = self.params['good_growth_rate']
        rate_difference = average_appreciation_rate - target_rate
        print(f"Average Appreciation Rate: {average_appreciation_rate:.4f}")
        print(f"Target Rate: {target_rate:.4f}")
        print(f"Difference from Target Rate: {rate_difference:.4f}")
        return self.good_price_history, self.interest_rate_history, self.interest_rate, self.fed_bond_history, self.bank_bond_history, self.bank_cash_history, self.transaction_history, self.inflation_history, self.composite_inflation_history, self.real_gdp_history

    def update_transaction_history(self, round, consumers, goods_categories, transaction_history, good_prices, buyers, sellers):
        buy_amount = jnp.sum(buyers * good_prices, axis=1)
        sell_amount = jnp.sum(sellers * good_prices, axis=1)

        for i, consumer in enumerate(consumers):
            consumer.capital -= buy_amount[i]
            consumer.capital += sell_amount[i]

        total_buy_amount = jnp.sum(buy_amount)
        total_sell_amount = jnp.sum(sell_amount)

        transaction_history.append((round, total_buy_amount, total_sell_amount))
        return transaction_history

    def update_good_price_history(self, round, good_prices, good_price_history):
        for i, price in enumerate(good_prices):
            good_price_history = good_price_history.at[round, i].set(price)
        return good_price_history