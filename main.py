import argparse
from qe_macro_calc.simulation.simulation import Simulation
from qe_macro_calc.utils.data_loader import load_parameters
from qe_macro_calc.plot import plot_results
import jax.random as jrandom

def main():
    parser = argparse.ArgumentParser(description='Run the macroeconomic simulation.')
    parser.add_argument('--params', type=str, required=True, help='Path to the parameters JSON file.')
    parser.add_argument('--strategy', type=str, required=False, help='Path to the custom strategy file.')
    args = parser.parse_args()

    params = load_parameters(args.params)
    key = jrandom.PRNGKey(0)

    if args.strategy:
        params['strategy'] = args.strategy
    else:
        # Default to base_strategy.py in the /strategy directory
        params['strategy'] = 'qe_macro_calc/strategy/base_strategy.py'

    simulation = Simulation(params, key)
    results = simulation.run()

    plot_results(results)

if __name__ == "__main__":
    main()