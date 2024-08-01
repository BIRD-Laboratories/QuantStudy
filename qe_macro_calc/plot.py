import matplotlib.pyplot as plt
import jax.numpy as jnp

def plot_results(results):
    good_price_history, interest_rate_history, interest_rate, fed_bond_history, bank_bond_history, bank_cash_history, transaction_history, inflation_history, composite_inflation_history, real_gdp_history = results
    rounds = jnp.arange(len(interest_rate_history))

    fig, axs = plt.subplots(7, 1, figsize=(10, 42))

    axs[0].plot(rounds, good_price_history, marker='o', linestyle='-', color='g')
    axs[0].set_title('Good Price Over Time')
    axs[0].set_xlabel('Round')
    axs[0].set_yscale('log')
    axs[0].set_ylabel('Good Price')
    axs[0].grid(True)

    axs[1].plot(rounds, interest_rate_history, marker='o', linestyle='-', color='m')
    axs[1].set_title('Interest Rate Over Time')
    axs[1].set_xlabel('Round')
    axs[1].set_ylabel('Interest Rate')
    axs[1].grid(True)

    axs[2].plot(rounds, fed_bond_history, marker='o', linestyle='-', color='c')
    axs[2].set_title('Federal Reserve Bond Ownership Over Time')
    axs[2].set_yscale('log')
    axs[2].set_xlabel('Round')
    axs[2].set_ylabel('Bonds Owned')
    axs[2].grid(True)

    for i in range(bank_bond_history.shape[1]):
        axs[3].plot(rounds, bank_bond_history[:, i], marker='o', linestyle='-', label=f'Bank {i}')
    axs[3].set_title('Bank Bond Ownership Over Time')
    axs[3].set_xlabel('Round')
    axs[3].set_ylabel('Bonds Owned')
    axs[3].legend()
    axs[3].grid(True)

    for i in range(bank_cash_history.shape[1]):
        axs[4].plot(rounds, bank_cash_history[:, i], marker='o', linestyle='-', label=f'Bank {i}')
    axs[4].set_title('Bank Cash Reserves Over Time')
    axs[4].set_xlabel('Round')
    axs[4].set_ylabel('Cash Reserves')
    axs[4].legend()
    axs[4].grid(True)

    axs[5].plot(rounds, inflation_history, marker='o', linestyle='-', color='b')
    axs[5].set_title('Inflation Over Time')
    axs[5].set_xlabel('Round')
    axs[5].set_ylabel('Inflation')
    axs[5].grid(True)

    axs[6].plot(rounds, composite_inflation_history, marker='o', linestyle='-', color='r')
    axs[6].set_title('Composite Inflation Over Time')
    axs[6].set_xlabel('Round')
    axs[6].set_ylabel('Composite Inflation')
    axs[6].grid(True)

    axs[7].plot(rounds, real_gdp_history, marker='o', linestyle='-', color='k')
    axs[7].set_title('Real GDP Growth Over Time')
    axs[7].set_xlabel('Round')
    axs[7].set_ylabel('Real GDP Growth')
    axs[7].grid(True)

    plt.savefig("economics.png")
    plt.show()