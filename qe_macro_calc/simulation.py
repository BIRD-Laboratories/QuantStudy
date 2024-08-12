import numpy as np
from qe_macro_calc.kernels import Kernels
from qe_macro_calc.metrics.gdp import GDP
from qe_macro_calc.utils.data_loader import DataLoader
from qe_macro_calc.metrics.vix import VIX

class Simulation:
    def __init__(self, params_file):
        self.params_file = params_file
        self.kernels = Kernels()
        self.data_loader = DataLoader(params_file)
        self.params = self.data_loader.params
        self.initialize_entities()
        self.vix = VIX()

    def initialize_entities(self):
        self.money_supply = float(self.params['initial_money_supply'])
        self.salary = self.params['salary']
        self.entities = self.data_loader.load_entities()
        self.interest_rate = self.params['initial_interest_rate']
        self.good_price_history = []
        self.interest_rate_history = []
        self.inflation_history = []
        self.gdp = GDP(self.params['goods_categories'])
        self.weights = np.array([good['weight'] for good in self.params['goods']], dtype=np.float32)

    def run_round(self, round, state):
        buy_amounts = np.zeros(len(self.entities['consumers']), dtype=np.float32)
        sell_amounts = np.zeros(len(self.entities['consumers']), dtype=np.float32)

        banks_array, consumers_array, goods_array = self.entities['banks'], self.entities['consumers'], self.entities['goods']

        interest_rate_array = np.array([self.interest_rate], dtype=np.float32)
        gdp_growth_array = np.array([state[6][round-1] if round > 0 else 0], dtype=np.float32)
        unemployment_rate_array = np.array([state[7][round-1] if round > 0 else 0], dtype=np.float32)
        interest_rate_adjustment_array = np.array([self.params['interest_rate_adjustment']], dtype=np.float32)
        bond_price_array = np.array([state[10][round-1] if round > 0 else self.params['initial_bond_price']], dtype=np.float32)
        money_supply_array = np.array([self.money_supply], dtype=np.float32)
        salary_array = np.array([self.salary], dtype=np.float32)

        last_prices = self.good_price_history[round - 1] if round > 0 else np.array([good['initial_price'] for good in self.params['goods']], dtype=np.float32)

        banks_array, consumers_array, goods_array, self.interest_rate, inflation, _, _, updated_money_supply, updated_salary = self.kernels.fused_kernel(
            banks_array, consumers_array, goods_array, interest_rate_array[0], buy_amounts, sell_amounts, gdp_growth_array[0], unemployment_rate_array[0], interest_rate_adjustment_array[0], bond_price_array[0], money_supply_array[0], salary_array[0], 10000, last_prices, round
        )

        state[0].append(self.interest_rate)
        state[1].append(np.sum(banks_array[:, 2]))
        state[2].append(inflation)
        state[3].append(updated_money_supply)
        state[4].append(updated_salary)
        state[5].append(0)  # Assuming gdp_growth is 0 for now
        state[6].append(0)  # Assuming unemployment_rate is 0 for now
        state[7].append(self.params['interest_rate_adjustment'])
        state[8].append(bond_price_array[0])
        state[9].append(0)  # Update bank bond buying
        state[10].append(0)  # Update bank bond selling

        self.good_price_history.append(goods_array[:, 2])

        # Calculate VIX
        volatility = np.std(state[2])
        vix = self.vix.calculate_vix(volatility)
        state[11].append(vix)

        return state

    def run(self):
        num_rounds = self.params['num_rounds']
        self.money_supply = float(self.params['initial_money_supply'])
        state = [
            [],  # Interest Rate
            [],  # Sum of total_money for banks
            [],  # Inflation
            [],  # Good Prices
            [],  # Money Supply
            [],  # Salary
            [],  # GDP Growth
            [],  # Unemployment Rate
            [],  # Interest Rate Adjustment
            [],  # Bond Price
            [],  # Bank Bond Buying
            [],  # Bank Bond Selling
            []   # VIX
        ]
        for round in range(num_rounds):
            state = self.run_round(round, state)

        return state