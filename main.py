import argparse
import pyopencl as cl
from qe_macro_calc.simulation import Simulation
from qe_macro_calc.utils.data_loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run the economic simulation.')
    parser.add_argument('--params_dir', type=str, default='parameters.json', help='Directory to load parameters from')
    args = parser.parse_args()

    # Load parameters from the specified directory
    data_loader = DataLoader(args.params_dir)
    params = data_loader.params

    # Check for OpenCL platforms
    platforms = cl.get_platforms()
    if not platforms:
        print("No OpenCL platforms found. Exiting.")
        return

    print("Available OpenCL platforms:")
    for i, platform in enumerate(platforms):
        print(f"{i}: {platform.name}")

    # Select the first platform
    selected_platform = platforms[0]
    print(f"Selected platform: {selected_platform.name}")

    # Get devices for the selected platform
    devices = selected_platform.get_devices()
    if not devices:
        print("No OpenCL devices found. Exiting.")
        return

    print("Available devices:")
    for i, device in enumerate(devices):
        print(f"{i}: {device.name}")

    # Select the first device
    selected_device = devices[0]
    print(f"Selected device: {selected_device.name}")

    # Create an OpenCL context
    ctx = cl.Context([selected_device])
    print("OpenCL context created successfully.")

    # Create an OpenCL command queue
    queue = cl.CommandQueue(ctx)

    # Initialize and run the simulation
    simulation = Simulation(args.params_dir, ctx, queue)
    state = simulation.run()

    # Print final state
    print("Simulation completed successfully.")
    print(f"Final Good Prices: {state[0][-1]}")
    print(f"Final Interest Rate: {state[2][-1]:.4f}")
    print(f"Final Fed bond amount: {state[3][-1]}")
    print(f"Final Inflation: {state[6][-1]:.4f}")
    print(f"Final Real GDP: {state[7][-1]:.4f}")
    print(f"Final Money Supply: {state[8][-1]:.4f}")
    print(f"Final Salary: {state[9][-1]:.4f}")
    print(f"Final Affordability Ratio: {state[15][-1]:.4f}")

    # Plotting
    num_rounds = params['num_rounds']
    rounds = np.arange(num_rounds)

    plt.figure(figsize=(15, 20))

    # Plot Good Prices Composite Index
    plt.subplot(4, 2, 1)
    composite_index = np.zeros(num_rounds)
    for round in range(num_rounds):
        composite_index[round] = np.sum(state[0][round] * simulation.weights)
    plt.plot(rounds, composite_index, label='Composite Index')
    plt.title('Good Prices Composite Index Over Time')
    plt.xlabel('Round')
    plt.ylabel('Composite Index')
    plt.legend()

    # Plot Interest Rate
    plt.subplot(4, 2, 2)
    plt.plot(rounds, state[2])
    plt.title('Interest Rate Over Time')
    plt.xlabel('Round')
    plt.ylabel('Interest Rate')

    # Plot Fed Bond Amount
    plt.subplot(4, 2, 3)
    plt.plot(rounds, state[3])
    plt.title('Fed Bond Amount Over Time')
    plt.xlabel('Round')
    plt.ylabel('Bond Amount')

    # Plot Inflation
    plt.subplot(4, 2, 4)
    plt.plot(rounds, state[6])
    plt.title('Inflation Over Time')
    plt.xlabel('Round')
    plt.ylabel('Inflation')

    # Plot Real GDP
    plt.subplot(4, 2, 5)
    plt.plot(rounds, state[7])
    plt.title('Real GDP Over Time')
    plt.xlabel('Round')
    plt.ylabel('Real GDP')

    # Plot Money Supply
    plt.subplot(4, 2, 6)
    plt.plot(rounds, state[8])
    plt.title('Money Supply Over Time')
    plt.xlabel('Round')
    plt.ylabel('Money Supply')

    # Plot Salaries
    plt.subplot(4, 2, 7)
    plt.plot(rounds, state[9])
    plt.title('Salaries Over Time')
    plt.xlabel('Round')
    plt.ylabel('Salary')

    # Plot Affordability Ratio
    plt.subplot(4, 2, 8)
    plt.plot(rounds, state[15])
    plt.title('Affordability Ratio Over Time')
    plt.xlabel('Round')
    plt.ylabel('Affordability Ratio')

    plt.tight_layout()
    plt.savefig("economics.png")
    plt.show()

if __name__ == "__main__":
    main()