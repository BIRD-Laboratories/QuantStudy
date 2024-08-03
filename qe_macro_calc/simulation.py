import numpy as np
from qe_macro_calc.kernels import Kernels
from qe_macro_calc.entities.entity import Bank, Consumer, Goods, Company
from qe_macro_calc.metrics.gdp import GDP
from qe_macro_calc.utils.data_loader import load_parameters

class Simulation:
    def __init__(self, params, ctx, queue):
        self.params = params
        self.ctx = ctx
        self.queue = queue
        self.kernels = Kernels(ctx, queue)
        self.data_loader = DataLoader(params)
        self.initialize_entities()

    def initialize_entities(self):
        self.entities = self.data_loader.load_entities()
        self.interest_rate = self.params['initial_interest_rate']
        self.good_price_history = np.zeros((self.params['num_rounds'], len(self.params['goods_categories'])))
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

        self.goods = Goods(self.params['goods_categories'], self.params['goods_weights'], self.params['initial_price_level'], self.params)
        self.gdp = GDP(self.params['goods_categories'])

    def run_round(self, round, state):
        recession_status = np.any(round == self.recession_rounds)

        buy_amounts = np.zeros(len(self.entities['consumers']), dtype=np.float32)
        sell_amounts = np.zeros(len(self.entities['consumers']), dtype=np.float32)

        self.entities['banks'], self.entities['consumers'], self.goods.prices, self.interest_rate, new_prices, inflation = self.kernels.fused_update(
            self.entities['banks'], self.entities['consumers'], self.goods.prices, self.interest_rate, buy_amounts, sell_amounts, 0, 0, self.params['interest_rate_adjustment'], recession_status, self.params['initial_bond_price']
        )

        state[0] = self.kernels.update_good_price_history_kernel(self.ctx, self.queue, round, new_prices, state[0])
        state[2][round] = self.interest_rate
        state[3][round] = np.sum(self.entities['banks'][:, 3])  # Sum of total_money for banks

        # Update good price history with adjusted prices
        state[0] = self.kernels.update_good_price_history_kernel(self.ctx, self.queue, round, new_prices, state[0])
        state[6][round] = inflation
        state[7][round] = self.gdp.calculate_gdp(new_prices, np.random.binomial(1, 0.5, size=len(self.entities['consumers'])))
        return state

    def run(self):
        state = [
            self.good_price_history,
            self.transaction_history,
            self.interest_rate_history,
            self.fed_bond_history,
            self.bank_bond_history,
            self.bank_cash_history,
            self.inflation_history,
            self.real_gdp_history
        ]

        for round in range(1, self.params['num_rounds']):
            state = self.run_round(round, state)

        print(f"Final Interest Rate: {state[2][self.params['num_rounds'] - 1]:.4f}")
        print(f"Fed bond amount: {state[3][self.params['num_rounds'] - 1]}")
        print(f"Final Good Prices: {state[0][self.params['num_rounds'] - 1]}")
        return state

# Example usage of pmap for parallel execution
def parallel_simulations(params_list, ctx, queue):
    simulations = [Simulation(params, ctx, queue) for params in params_list]
    results = [sim.run() for sim in simulations]
    return results
