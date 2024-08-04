import numpy as np
from qe_macro_calc.utils.data_loader import DataLoader
from qe_macro_calc.kernels import Kernels

class Simulation:
    def __init__(self, params_file, ctx, queue):
        self.data_loader = DataLoader(params_file)
        self.params = self.data_loader.params
        self.ctx = ctx
        self.queue = queue
        self.kernels = Kernels(self.ctx, self.queue)
        self.initialize_entities()
        self.initialize_state()

    def initialize_entities(self):
        self.entities = self.data_loader.load_entities()

    def initialize_state(self):
        num_rounds = self.params['num_rounds']
        num_goods = len(self.entities['goods'])
        self.state = {
            'interest_rate': np.zeros(num_rounds, dtype=np.float32),
            'total_bank_capital': np.zeros(num_rounds, dtype=np.float32),
            'inflation': np.zeros(num_rounds, dtype=np.float32),
            'money_supply': np.zeros(num_rounds, dtype=np.float32),
            'salary': np.zeros(num_rounds, dtype=np.float32),
            'gdp_growth': np.zeros(num_rounds, dtype=np.float32),
            'unemployment_rate': np.zeros(num_rounds, dtype=np.float32),
            'interest_rate_adjustment': np.zeros(num_rounds, dtype=np.float32),
            'recession_status': np.zeros(num_rounds, dtype=np.int32),
            'bond_price': np.zeros(num_rounds, dtype=np.float32),
            'affordability_ratio': np.zeros(num_rounds, dtype=np.float32),
            'bank_bond_buying': np.zeros(num_rounds, dtype=np.float32),
            'bank_bond_selling': np.zeros(num_rounds, dtype=np.float32),
            'historical_inflation': np.zeros(num_rounds, dtype=np.float32),
            'previous_prices': np.array(self.entities['goods'], dtype=np.float32)  # Initialize with initial prices
        }
        self.state['interest_rate'][0] = self.params['initial_interest_rate']
        self.state['money_supply'][0] = self.params['initial_money_supply']
        self.state['salary'][0] = self.params['salary']
        self.state['bond_price'][0] = self.params['initial_bond_price']
        self.state['interest_rate_adjustment'][0] = self.params['interest_rate_adjustment']

    def run(self):
        num_rounds = self.params['num_rounds']
        for round in range(1, num_rounds):
            self.run_round(round)
        return self.state

    def run_round(self, round):
        recession_status = False

        buy_amounts = np.zeros(len(self.entities['consumers']), dtype=np.float32)
        sell_amounts = np.zeros(len(self.entities['consumers']), dtype=np.float32)

        banks_array, consumers_array, goods_array = self.entities['banks'], self.entities['consumers'], self.entities['goods']

        interest_rate = self.state['interest_rate'][round - 1]
        money_supply = self.state['money_supply'][round - 1]
        salary = self.state['salary'][round - 1]
        bond_price = self.state['bond_price'][round - 1]
        historical_inflation = self.state['historical_inflation']
        interest_rate_adjustment = self.state['interest_rate_adjustment'][round - 1]
        previous_prices = self.state['previous_prices']

        result = self.kernels.fused_kernel(
            banks_array, consumers_array, goods_array, interest_rate, buy_amounts, sell_amounts,
            self.state['gdp_growth'][round - 1], self.state['unemployment_rate'][round - 1],
            interest_rate_adjustment, recession_status, bond_price, money_supply, salary, 10000, historical_inflation, previous_prices, round
        )

        self.state['interest_rate'][round] = result[3]
        self.state['total_bank_capital'][round] = np.sum(result[0][:, 2])
        self.state['inflation'][round] = result[4]
        self.state['money_supply'][round] = result[7]
        self.state['salary'][round] = result[8]  # Update salary based on inflation
        self.state['gdp_growth'][round] = 0  # Placeholder
        self.state['unemployment_rate'][round] = 0  # Placeholder
        self.state['interest_rate_adjustment'][round] = interest_rate_adjustment
        self.state['recession_status'][round] = recession_status
        self.state['bond_price'][round] = bond_price
        self.state['affordability_ratio'][round] = result[9]
        self.state['bank_bond_buying'][round] = result[5]
        self.state['bank_bond_selling'][round] = result[6]
        self.state['historical_inflation'] = result[10]
        self.state['previous_prices'] = result[11]

        self.entities['banks'] = result[0]
        self.entities['consumers'] = result[1]
        self.entities['goods'] = result[2]