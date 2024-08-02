import matplotlib.pyplot as plt
import jax.numpy as jnp

def plot_results(results):
    good_price_history, transaction_history, interest_rate_history, fed_bond_history, bank_bond_history, bank_cash_history, inflation_history = results
    rounds = jnp.arange(len(interest_rate_history))

    fig, axs = plt.subplots(5, 1, figsize=(10, 70))

    axs[0].plot(rounds, good_price_history, marker='o', linestyle='-', color='g')
    axs[0].set_title('Good Price Over Time')
    axs[0].set_xlabel('Round')
    axs[0].set_ylabel('Good Price')
    axs[0].grid(True)

    axs[1].plot(rounds, interest_rate_history, marker='o', linestyle='-', color='m')
    axs[1].set_title('Interest Rate Over Time')
    axs[1].set_xlabel('Round')
    axs[1].set_ylabel('Interest Rate')
    axs[1].grid(True)

    axs[2].plot(rounds, fed_bond_history, marker='o', linestyle='-', color='c')
    axs[2].set_title('Federal Reserve Bond Ownership Over Time')
    axs[2].set_xlabel('Round')
    axs[2].set_ylabel('Bonds Owned')
    axs[2].grid(True)

    axs[3].plot(rounds, inflation_history, marker='o', linestyle='-', color='r')
    axs[3].set_title('Inflation Over Time')
    axs[3].set_xlabel('Round')
    axs[3].set_ylabel('Inflation Rate')
    axs[3].grid(True)


    axs[4].plot(rounds, bank_bond_history, marker='o', linestyle='-', color='b')
    axs[4].set_title('Bank Bond Ownership Over Time')
    axs[4].set_xlabel('Round')
    axs[4].set_ylabel('Bonds Owned')
    axs[4].grid(True)


    plt.tight_layout()
    plt.savefig("economics.png")
    plt.show()
