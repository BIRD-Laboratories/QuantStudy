import jax.numpy as jnp
import jax.random as jrandom

class Company:
    def __init__(self, category, bank, initial_employees):
        self.category = category
        self.bank = bank
        self.employees = jnp.array(initial_employees, dtype=jnp.int32)
        self.loan_amount = jnp.array(0, dtype=jnp.float32)

    def borrow(self, amount, interest_rate):
        borrow_condition = interest_rate < 0.05
        self.loan_amount += jnp.where(borrow_condition, amount, 0)
        self.employees += jnp.where(borrow_condition, amount // 1000, -self.employees // 10)
        self.bank.lend(jnp.where(borrow_condition, amount, 0))

    def repay(self, amount):
        self.loan_amount -= amount
        self.bank.receive_payment(amount)

    def produce(self, goods):
        # Base production logic
        pass

class FoodCompany(Company):
    def produce(self, goods):
        # Specific production logic for food
        food_price = goods.prices[goods.categories == 'Food']
        production_cost = food_price * 0.5  # Example cost
        return jnp.float32(production_cost)

class EnergyCompany(Company):
    def produce(self, goods):
        # Specific production logic for energy
        energy_price = goods.prices[goods.categories == 'Energy']
        production_cost = energy_price * 0.7  # Example cost
        return jnp.float32(production_cost)

class HousingCompany(Company):
    def produce(self, goods):
        # Specific production logic for housing
        housing_price = goods.prices[goods.categories == 'Housing']
        production_cost = housing_price * 1.2  # Example cost
        return jnp.float32(production_cost)