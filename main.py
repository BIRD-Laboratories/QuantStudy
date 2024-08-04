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
    print(f"Final Interest Rate: {state['interest_rate'][-1]:.4f}")
    print(f"Final Affordability Ratio: {state['affordability_ratio'][-1]:.4f}")

    # Plotting
    num_rounds = params['num_rounds']
    rounds = np.arange(num_rounds)

    plt.figure(figsize=(15, 20))

    # Plot Interest Rate
    plt.subplot(4, 2, 1)
    plt.plot(rounds, state['interest_rate'])
    plt.title('Interest Rate Over Time')
    plt.xlabel('Round')
    plt.ylabel('Interest Rate')

    # Plot Money Supply
    plt.subplot(4, 2, 2)
    plt.plot(rounds, state['money_supply'])
    plt.title('Money Supply Over Time')
    plt.xlabel('Round')
    plt.ylabel('Money Supply')

    # Plot Salaries
    plt.subplot(4, 2, 3)
    plt.plot(rounds, state['salary'])
    plt.title('Salaries Over Time')
    plt.xlabel('Round')
    plt.ylabel('Salary')

    # Plot Affordability Ratio
    plt.subplot(4, 2, 4)
    plt.plot(rounds, state['affordability_ratio'])
    plt.title('Affordability Ratio Over Time')
    plt.xlabel('Round')
    plt.ylabel('Affordability Ratio')

    # Plot Inflation
    plt.subplot(4, 2, 5)
    plt.plot(rounds, state['inflation'])
    plt.title('Inflation Over Time')
    plt.xlabel('Round')
    plt.ylabel('Inflation')

    # Plot Bank Bond Buying
    plt.subplot(4, 2, 6)
    plt.plot(rounds, state['bank_bond_buying'])
    plt.title('Bank Bond Buying Over Time')
    plt.xlabel('Round')
    plt.ylabel('Bank Bond Buying')

    # Plot Bank Bond Selling
    plt.subplot(4, 2, 7)
    plt.plot(rounds, state['bank_bond_selling'])
    plt.title('Bank Bond Selling Over Time')
    plt.xlabel('Round')
    plt.ylabel('Bank Bond Selling')

    plt.tight_layout()
    plt.savefig("economics.png")
    plt.show()

if __name__ == "__main__":
    main()