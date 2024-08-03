import numpy as np
from qe_macro_calc.kernels import Kernels
from qe_macro_calc.metrics.gdp import GDP
from qe_macro_calc.utils.data_loader import DataLoader

class Simulation:
    def __init__(self, params_file, ctx, queue):
        self.params_file = params_file
        self.ctx = ctx
        self.queue = queue
        self.kernels = Kernels(ctx, queue)
        self.data_loader = DataLoader(params_file)
        self.params = self.data_loader.params
        self.initialize_entities()

    def initialize_entities(self):
        self.money_supply = self.params['initial_money_supply']
        self.salary = self.params['salary']
        self.entities = self.data_loader.load_entities()
        self.interest_rate = self.params['initial_interest_rate']
        self.good_price_history = np.zeros((self.params['num_rounds'], len(self.params['goods'])))
        self.interest_rate_history = np.zeros(self.params['num_rounds'])
        self.interest_rate_history[0] = self.params['initial_interest_rate']
        self.transaction_history = np.zeros(3)  # Initialize with fixed length
        self.inflation_history = np.zeros(self.params['num_rounds'])  # Initialize as a NumPy array
        self.composite_inflation_history = np.zeros(self.params['num_rounds'])  # Initialize as a NumPy array
        self.real_gdp_history = np.zeros(self.params['num_rounds'])  # Initialize as a NumPy array

        self.recession_rounds = np.random.choice(np.arange(1, self.params['num_rounds'] - self.params['recession_duration_max']), self.params['recessions'], replace=False)
        self.recession_durations = np.random.randint(self.params['recession_duration_min'], self.params['recession_duration_max'] + 1, self.params['recessions'])

        self.fed_bond_history = np.zeros(self.params['num_rounds'])
        self.bank_bond_history = np.zeros((self.params['num_rounds'], len(self.entities['banks'])))
        self.bank_cash_history = np.zeros((self.params['num_rounds'], len(self.entities['banks'])))

        self.gdp = GDP(self.params['goods_categories'])

        # Ensure weights is a numpy array
        self.weights = np.array([good['weight'] for good in self.params['goods']], dtype=np.float32)

    def run_round(self, round, state):
        recession_status = np.any(round == self.recession_rounds)

        buy_amounts = np.zeros(len(self.entities['consumers']), dtype=np.float32)
        sell_amounts = np.zeros(len(self.entities['consumers']), dtype=np.float32)

        banks_array, consumers_array, goods_array = self.entities['banks'], self.entities['consumers'], self.entities['goods']

        # Convert scalar values to numpy arrays
        interest_rate_array = np.array([self.interest_rate], dtype=np.float32)
        gdp_growth_array = np.array([state[10][round-1] if round > 0 else 0], dtype=np.float32)  # GDP Growth from previous round
        unemployment_rate_array = np.array([state[11][round-1] if round > 0 else 0], dtype=np.float32)  # Unemployment Rate from previous round
        interest_rate_adjustment_array = np.array([self.params['interest_rate_adjustment']], dtype=np.float32)
        recession_status_array = np.array([recession_status], dtype=np.int32)
        bond_price_array = np.array([state[14][round-1] if round > 0 else self.params['initial_bond_price']], dtype=np.float32)  # Bond Price from previous round
        money_supply_array = np.array([self.money_supply], dtype=np.float32)
        salary_array = np.array([self.salary], dtype=np.float32)

        # Call fused_kernel for interest rate adjustments and QE
        new_prices, inflation, bank_bond_buying, bank_bond_selling, updated_money_supply, updated_salary, affordability_ratio = self.kernels.fused_kernel(
            banks_array, consumers_array, goods_array, interest_rate_array, buy_amounts, sell_amounts, gdp_growth_array, unemployment_rate_array, interest_rate_adjustment_array, recession_status_array, bond_price_array, self.weights, money_supply_array, salary_array
        )
        self.money_supply *= 1.04

        # Increase salary by 2% each round
        self.salary *= 1.02
        # Ensure new_prices has the correct shape
        if new_prices.shape != (len(self.params['goods']),):
            raise ValueError(f"Expected new_prices to have shape ({len(self.params['goods'])},), but got {new_prices.shape}")

        # Update state with new prices and inflation
        state[0][round] = new_prices
        state[2][round] = self.interest_rate
        state[3][round] = np.sum(self.entities['banks'][:, 2])  # Sum of total_money for banks
        state[6][round] = inflation
        state[7][round] = self.gdp.calculate_gdp(new_prices, np.random.binomial(1, 0.5, size=len(self.entities['consumers'])))
        state[8][round] = self.money_supply
        state[9][round] = self.salary
        state[10][round] = 0  # Assuming gdp_growth is 0 for now
        state[11][round] = 0  # Assuming unemployment_rate is 0 for now
        state[12][round] = self.params['interest_rate_adjustment']
        state[13][round] = recession_status
        state[14][round] = bond_price_array[0]  # Update bond price
        state[15][round] = affordability_ratio
        # Update bank bond buying and selling history
        state[4][round] = bank_bond_buying
        state[5][round] = bank_bond_selling

        return state

    def run(self):
        num_rounds = self.params['num_rounds']
        state = [
            np.zeros((num_rounds, len(self.params['goods'])), dtype=np.float32),  # Good Prices
            np.zeros((num_rounds, self.params['num_consumers']), dtype=np.float32),  # Consumers
            np.zeros(num_rounds, dtype=np.float32),  # Interest Rate
            np.zeros(num_rounds, dtype=np.float32),  # Fed Bond Amount
            np.zeros((num_rounds, self.params['num_banks']), dtype=np.float32),  # Bank Bond Buying
            np.zeros((num_rounds, self.params['num_banks']), dtype=np.float32),  # Bank Bond Selling
            np.zeros(num_rounds, dtype=np.float32),  # Inflation
            np.zeros(num_rounds, dtype=np.float32),  # Real GDP
            np.zeros(num_rounds, dtype=np.float32),  # Money Supply
            np.zeros(num_rounds, dtype=np.float32),  # Salary
            np.zeros(num_rounds, dtype=np.float32),  # GDP Growth
            np.zeros(num_rounds, dtype=np.float32),  # Unemployment Rate
            np.zeros(num_rounds, dtype=np.float32),  # Interest Rate Adjustment
            np.zeros(num_rounds, dtype=np.int32),  # Recession Status
            np.zeros(num_rounds, dtype=np.float32),  # Bond Price
            np.zeros(num_rounds, dtype=np.float32),
        ]

        for round in range(num_rounds):
            state = self.run_round(round, state)

        return state