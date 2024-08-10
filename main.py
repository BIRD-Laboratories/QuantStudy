import argparse
import numpy as np
import matplotlib.pyplot as plt
from qe_macro_calc.simulation import Simulation
from qe_macro_calc.utils.data_loader import DataLoader

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run the economic simulation.')
    parser.add_argument('--params_dir', type=str, default='parameters.json', help='Directory to load parameters from')
    args = parser.parse_args()

    # Load parameters from the specified directory
    data_loader = DataLoader(args.params_dir)
    params = data_loader.params

    # Initialize and run the simulation
    simulation = Simulation(args.params_dir)
    state = simulation.run()

    # Print final state
    print("Simulation completed successfully.")

    # Plotting
    num_rounds = params['num_rounds']
    rounds = np.arange(num_rounds)

    plt.figure(figsize=(15, 20))

    # Plot Interest Rate
    plt.subplot(4, 2, 1)
    plt.plot(rounds, state[0])
    plt.title('Interest Rate Over Time')
    plt.xlabel('Round')
    plt.ylabel('Interest Rate')

    # Plot Money Supply
    plt.subplot(4, 2, 2)
    plt.plot(rounds, state[3])
    plt.title('Money Supply Over Time')
    plt.xlabel('Round')
    plt.ylabel('Money Supply')

    # Plot Salaries
    plt.subplot(4, 2, 3)
    plt.plot(rounds, state[4])
    plt.title('Salaries Over Time')
    plt.xlabel('Round')
    plt.ylabel('Salary')

    # Plot Inflation
    plt.subplot(4, 2, 4)
    plt.plot(rounds, state[2])
    plt.title('Inflation Over Time')
    plt.xlabel('Round')
    plt.ylabel('Inflation')

    # Plot Total Bank Money
    plt.subplot(4, 2, 5)
    plt.plot(rounds, state[1])
    plt.title('Total Bank Money Over Time')
    plt.xlabel('Round')
    plt.ylabel('Total Bank Money')

    plt.tight_layout()
    plt.savefig("economics.png")
    plt.show()

if __name__ == "__main__":
    main()