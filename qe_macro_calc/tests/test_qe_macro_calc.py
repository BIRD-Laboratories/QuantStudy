import unittest
import numpy as np
import logging
from qe_macro_calc.utils.data_loader import DataLoader
from qe_macro_calc.simulation import Simulation
from qe_macro_calc.kernels import Kernels

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        logging.debug("Setting up TestDataLoader")
        self.data_loader = DataLoader('parameters.json')

    def test_load_parameters(self):
        logging.debug("Testing load_parameters")
        params = self.data_loader.load_parameters()
        self.assertIsInstance(params, dict)
        self.assertIn('num_rounds', params)

    def test_load_entities(self):
        logging.debug("Testing load_entities")
        entities = self.data_loader.load_entities()
        self.assertIsInstance(entities, dict)
        self.assertIn('banks', entities)
        self.assertIn('consumers', entities)
        self.assertIn('goods', entities)
        self.assertIsInstance(entities['banks'], np.ndarray)

class TestKernels(unittest.TestCase):
    def setUp(self):
        logging.debug("Setting up TestKernels")
        self.kernels = Kernels()

    def test_fused_kernel(self):
        logging.debug("Testing fused_kernel")
        banks = np.array([[0, 1000, 100, 0.05, 1, 0.5]], dtype=np.float32)
        consumers = np.array([[0, 1000, 100, 30, 5000, 2000, 0.5]], dtype=np.float32)
        goods = np.array([[0, 1, 100]], dtype=np.float32)
        interest_rate = 0.05
        buy_amounts = np.array([0], dtype=np.float32)
        sell_amounts = np.array([0], dtype=np.float32)
        gdp_growth = np.array([0], dtype=np.float32)
        unemployment_rate = np.array([0], dtype=np.float32)
        interest_rate_adjustment = 0.01
        bond_price = 100
        money_supply = 10000
        salary = 5000
        money_supply_increment = 100
        last_prices = np.array([100], dtype=np.float32)

        result = self.kernels.fused_kernel(
            banks, consumers, goods, interest_rate, buy_amounts, sell_amounts, gdp_growth, unemployment_rate,
            interest_rate_adjustment, bond_price, money_supply, salary, money_supply_increment, last_prices, 0, 10
        )

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 9)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsInstance(result[1], np.ndarray)
        self.assertIsInstance(result[2], np.ndarray)
        self.assertIsInstance(result[3], float)
        self.assertIsInstance(result[4], float)
        self.assertIsInstance(result[5], float)
        self.assertIsInstance(result[6], float)
        self.assertIsInstance(result[7], float)
        self.assertIsInstance(result[8], float)

class TestSimulation(unittest.TestCase):
    def setUp(self):
        logging.debug("Setting up TestSimulation")
        self.simulation = Simulation('parameters.json')

    def test_initialize_entities(self):
        logging.debug("Testing initialize_entities")
        self.assertIsInstance(self.simulation.money_supply, float)
        self.assertIsInstance(self.simulation.salary, float)
        self.assertIsInstance(self.simulation.entities, dict)
        self.assertIsInstance(self.simulation.interest_rate, float)
        self.assertIsInstance(self.simulation.good_price_history, list)
        self.assertIsInstance(self.simulation.interest_rate_history, list)
        self.assertIsInstance(self.simulation.inflation_history, list)
        self.assertIsInstance(self.simulation.gdp, object)
        self.assertIsInstance(self.simulation.weights, np.ndarray)
        self.assertIsInstance(self.simulation.vix, object)

    def test_run_round(self):
        logging.debug("Testing run_round")
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
        result = self.simulation.run_round(0, state)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 12)
        for item in result:
            self.assertIsInstance(item, list)

    def test_run(self):
        logging.debug("Testing run")
        result = self.simulation.run()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 12)
        for item in result:
            self.assertIsInstance(item, list)

if __name__ == '__main__':
    unittest.main()