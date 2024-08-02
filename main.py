import argparse
from qe_macro_calc.simulation.simulation import Simulation
from qe_macro_calc.utils.data_loader import load_parameters
from qe_macro_calc.plot import plot_results
import jax.random as jrandom
import logging
logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description='Run the macroeconomic simulation.')
    parser.add_argument('--params', type=str, required=True, help='Path to the parameters JSON file.')
    args = parser.parse_args()

    params = load_parameters(args.params)
    key = jrandom.PRNGKey(0)

    simulation = Simulation(params, key)
    results = simulation.run()

    plot_results(results)

if __name__ == "__main__":
    main()