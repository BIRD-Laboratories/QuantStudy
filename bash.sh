#!/bin/bash

# Create directories
mkdir -p qe_macro_calc/entities
mkdir -p qe_macro_calc/market
mkdir -p qe_macro_calc/policy
mkdir -p qe_macro_calc/strategy
mkdir -p qe_macro_calc/simulation
mkdir -p qe_macro_calc/utils
mkdir -p qe_macro_calc/metrics
mkdir -p strategies

# Create __init__.py files
touch qe_macro_calc/__init__.py
touch qe_macro_calc/entities/__init__.py
touch qe_macro_calc/market/__init__.py
touch qe_macro_calc/policy/__init__.py
touch qe_macro_calc/strategy/__init__.py
touch qe_macro_calc/simulation/__init__.py
touch qe_macro_calc/utils/__init__.py
touch qe_macro_calc/metrics/__init__.py

# Create entity.py
cat <<EOL > qe_macro_calc/entities/entity.py
class Entity:
    def __init__(self, name, initial_capital, max_bid):
        self.name = name
        self.capital = initial_capital
        self.max_bid = max_bid
EOL

# Create bank.py
cat <<EOL > qe_macro_calc/entities/bank.py
from qe_macro_calc.strategy.base_strategy import BaseStrategy

class Bank(Entity):
    def __init__(self, name, initial_capital, max_bid, initial_interest_rate_markup, can_sell_bonds, risk_appetite, strategy: BaseStrategy):
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
EOL

# Create consumer.py
cat <<EOL > qe_macro_calc/entities/consumer.py
from qe_macro_calc.strategy.base_strategy import BaseStrategy

class Consumer(Entity):
    def __init__(self, initial_capital, max_bid, age, salary, fixed_expenses, credit_score, strategy: BaseStrategy):
        super().__init__("", initial_capital, max_bid)
        self.age = age
        self.salary = salary
        self.fixed_expenses = fixed_expenses
        self.credit_score = credit_score
        self.strategy = strategy

    def calculate_purchase_probabilities(self, good_prices, interest_rate):
        return self.strategy.calculate_purchase_probabilities(self, good_prices, interest_rate)
EOL

# Create goods.py
cat <<EOL > qe_macro_calc/entities/goods.py
import jax.random as jrandom
import jax

class Goods:
    def __init__(self, categories, weights, initial_price_level):
        if len(categories) != len(weights):
            raise ValueError("Categories and weights must have the same length.")

        self.categories = categories
        self.weights = {cat: weight for cat, weight in zip(categories, weights)}
        self.prices = {cat: weight * initial_price_level for cat, weight in zip(categories, weights)}

    def update_prices(self, interest_rate, recession_status):
        bond_rate = 0.007 + interest_rate
        for category in self.categories:
            if recession_status == "in_progress":
                self.prices[category] *= (1 - PARAMETERS['recession_severity'])
            elif category == 'Housing':
                self.prices[category] *= (1 + bond_rate)
            else:
                self.prices[category] *= (1 + jrandom.uniform(jax.random.PRNGKey(0), (), minval=-0.01, maxval=0.03))

    def calculate_inflation(self):
        total_weight = sum(self.weights.values())
        weighted_prices = [self.prices[cat] * self.weights[cat] for cat in self.categories]
        return sum(weighted_prices) / total_weight

    def get_price(self, category):
        return self.prices.get(category, None)

    def get_all_prices(self):
        return self.prices
EOL

# Create market.py
cat <<EOL > qe_macro_calc/market/market.py
class Market:
    def __init__(self, liquidity_factor):
        self.liquidity_factor = liquidity_factor

    def impact_asset_prices(self, assets, interest_rate):
        for asset in assets:
            asset.price *= (1 + interest_rate * self.liquidity_factor)
EOL

# Create market_dynamics.py
cat <<EOL > qe_macro_calc/market/market_dynamics.py
import jax.numpy as jnp
import jax.random as jrandom

class MarketDynamics:
    def __init__(self, goods, consumers, interest_rate, recession_status, key):
        self.goods = goods
        self.consumers = consumers
        self.interest_rate = interest_rate
        self.recession_status = recession_status
        self.key = key

    def update_market_dynamics(self):
        self.goods.update_prices(self.interest_rate, self.recession_status)
        inflation = self.goods.calculate_inflation()

        good_prices = jnp.array(list(self.goods.prices.values()))
        purchase_probs = jnp.array([consumer.calculate_purchase_probabilities(good_prices, self.interest_rate) for consumer in self.consumers])

        buyers = jrandom.bernoulli(self.key, purchase_probs, (len(self.consumers), len(self.goods.categories)))
        sellers = jrandom.bernoulli(self.key, 0.1, (len(self.consumers), len(self.goods.categories)))

        bid_ask_spread = jnp.sum(buyers, axis=0) - jnp.sum(sellers, axis=0)
        for i, category in enumerate(self.goods.categories):
            self.goods.prices[category] *= (1 + 0.01 * bid_ask_spread[i] / len(self.consumers))

        return list(self.goods.prices.values()), buyers, sellers, inflation
EOL

# Create monetary_policy.py
cat <<EOL > qe_macro_calc/policy/monetary_policy.py
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
EOL

# Create base_strategy.py
cat <<EOL > qe_macro_calc/strategy/base_strategy.py
class BaseStrategy:
    def adjust_interest_rate(self, inflation, gdp_growth, unemployment_rate, interest_rate, interest_rate_adjustment, recession_status):
        raise NotImplementedError

    def implement_quantitative_easing(self, bond_price, recession_status):
        raise NotImplementedError

    def calculate_purchase_probabilities(self, consumer, good_prices, interest_rate):
        raise NotImplementedError

    def lend_to_consumers(self, bank, consumers, good_price, margin_requirement, base_interest_rate):
        raise NotImplementedError
EOL

# Create simulation.py
cat <<EOL > qe_macro_calc/simulation/simulation.py
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

        strategy = self.load_strategy(self.params.get('strategy'))

        self.monetary_policy = MonetaryPolicy(self.params, strategy)

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
        good_prices = jrandom.normal(key, (size,), mean=initial_good_price, stddev=good_price_std_dev)
        good_prices = jnp.clip(good_prices, 0, None)
        return ages, salaries, fixed_expenses, wage_increases, good_prices

    def run(self):
        for round in range(1, self.params['num_rounds']):
            if round in self.recession_rounds:
                recession_status = 'in_progress'
            else:
                recession_status = 'over'
            self.key, subkey = jrandom.split(self.key)
            market_dynamics = MarketDynamics(self.goods, self.consumers, self.interest_rate, recession_status, subkey)
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

    def load_strategy(self, strategy_path):
        if strategy_path:
            spec = importlib.util.spec_from_file_location("custom_strategy", strategy_path)
            custom_strategy = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_strategy)
            strategy_class = getattr(custom_strategy, "CustomStrategy")
            return strategy_class()
        else:
            strategy_module = __import__("custom_strategy", strategy_path)
            custom_strategy = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_strategy)
            strategy_class = getattr(custom_strategy, "CustomStrategy")
            return strategy_class()
        else:
            strategy_module = __import__(f"strategies.{self.params['strategy']}", fromlist=[self.params['strategy']])
            strategy_class = getattr(strategy_module, self.params['strategy'])
            return strategy_class()