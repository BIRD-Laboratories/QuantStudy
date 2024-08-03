import pyopencl as cl
from qe_macro_calc.simulation import Simulation
import numpy as np

def main():
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

    # Define simulation parameters
    params = {
        'num_rounds': 100,
        'goods_categories': ['Food', 'Clothing', 'Electronics'],
        'goods_weights': [0.4, 0.3, 0.3],
        'initial_price_level': 100.0,
        'initial_interest_rate': 0.05,
        'interest_rate_adjustment': 0.01,
        'initial_bond_price': 1000.0,
        'bond_yield': 0.03,
        'num_banks': 5,
        'num_consumers': 100,
        'num_companies': 10,
        'recessions': 2,
        'recession_duration_min': 5,
        'recession_duration_max': 10,
        'size': 100,
        'banks_can_sell_bonds': [True] * 5
    }

    # Initialize and run the simulation
    simulation = Simulation(params, ctx, queue)
    state = simulation.run()

    # Print final state
    print("Simulation completed successfully.")
    print(f"Final Good Prices: {state[0][-1]}")
    print(f"Final Interest Rate: {state[2][-1]:.4f}")
    print(f"Final Fed bond amount: {state[3][-1]}")
    print(f"Final Inflation: {state[6][-1]:.4f}")
    print(f"Final Real GDP: {state[7][-1]:.4f}")

if __name__ == "__main__":
    main()
