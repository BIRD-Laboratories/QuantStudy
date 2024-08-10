import numpy as np
from qe_macro_calc.kernels import Kernels
from qe_macro_calc.metrics.gdp import GDP
from qe_macro_calc.utils.data_loader import DataLoader

class Simulation:
    def __init__(self, params_file):
        self.params_file = params_file
        self.kernels = Kernels()
        self.data_loader = DataLoader(params_file)
        self.params = self.data_loader.params
        self.initialize_entities()

    def initialize_entities(self):
        self.money_supply = self.params['initial_money_supply']
        self.salary = self.params['salary']
        self.entities = self.data_loader.load_entities()
        self.interest_rate = self.params['initial_interest_rate']
        self.good_price_history = []
        self.interest_rate_history = []
        self.inflation_history = []
        self.gdp = GDP(self.params['goods_categories'])

        # Ensure weights is a numpy array
        self.weights = np.array([good['weight'] for good in self.params['goods']], dtype=np.float32)

    def run_round(self, round, state):
        buy_amounts = np.zeros(len(self.entities['consumers']), dtype=np.float32)
        sell_amounts = np.zeros(len(self.entities['consumers']), dtype=np.float32)

        banks_array, consumers_array, goods_array = self.entities['banks'], self.entities['consumers'], self.entities['goods']

        # Convert scalar values to numpy arrays
        interest_rate_array = np.array([self.interest_rate], dtype=np.float32)
        gdp_growth_array = np.array([state[6][round-1] if round > 0 else 0], dtype=np.float32)  # GDP Growth from previous round
        unemployment_rate_array = np.array([state[7][round-1] if round > 0 else 0], dtype=np.float32)  # Unemployment Rate from previous round
        interest_rate_adjustment_array = np.array([self.params['interest_rate_adjustment']], dtype=np.float32)
        bond_price_array = np.array([state[10][round-1] if round > 0 else self.params['initial_bond_price']], dtype=np.float32)  # Bond Price from previous round
        money_supply_array = np.array([self.money_supply], dtype=np.float32)
        salary_array = np.array([self.salary], dtype=np.float32)

        # Get last round's prices
        last_prices = self.good_price_history[round - 1] if round > 0 else np.array([good['initial_price'] for good in self.params['goods']], dtype=np.float32)

        # Call fused_kernel for interest rate adjustments and QE
        banks, consumers, goods, interest_rate, inflation, bank_bond_buying, bank_bond_selling, updated_money_supply, updated_salary = self.kernels.fused_kernel(
            banks_array, consumers_array, goods_array, interest_rate_array, buy_amounts, sell_amounts, gdp_growth_array, unemployment_rate_array, interest_rate_adjustment_array, bond_price_array, money_supply_array, salary_array, 10000, last_prices
        )

        # Update state with new prices and inflation
        state[0].append(interest_rate)  # Update interest rate
        state[1].append(np.sum(self.entities['banks'][:, 2]))  # Sum of total_money for banks
        state[2].append(inflation)  # Update inflation
        state[3].append(updated_money_supply)  # Update money supply
        state[4].append(updated_salary)  # Update salary
        state[5].append(0)  # Assuming gdp_growth is 0 for now
        state[6].append(0)  # Assuming unemployment_rate is 0 for now
        state[7].append(self.params['interest_rate_adjustment'])
        state[8].append(bond_price_array[0])  # Update bond price
        state[9].append(bank_bond_buying)  # Update bank bond buying
        state[10].append(bank_bond_selling)  # Update bank bond selling

        # Update good price history
        self.good_price_history.append(goods_array[:, 2])

        return state

    def run(self):
        num_rounds = self.params['num_rounds']
        self.money_supply = self.params['initial_money_supply']
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
        ]
        for round in range(num_rounds):
            state = self.run_round(round, state)

        return state