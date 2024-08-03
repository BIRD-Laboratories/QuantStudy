import json
import numpy as np

class DataLoader:
    def __init__(self, params_file):
        self.params_file = params_file
        self.params = self.load_parameters()
        self.category_to_id = {category: i for i, category in enumerate(self.params['goods_categories'])}

    def load_parameters(self):
        with open(self.params_file, 'r') as file:
            params = json.load(file)
        return params

    def load_entities(self):
        banks = self.load_banks()
        consumers = self.load_consumers()
        companies = self.load_companies()
        goods = self.load_goods()

        return {
            'banks': np.array(banks, dtype=np.float32),
            'consumers': np.array(consumers, dtype=np.float32),
            'companies': np.array(companies, dtype=np.float32),
            'goods': np.array(goods, dtype=np.float32)
        }

    def load_banks(self):
        banks = []
        for i in range(self.params['num_banks']):
            bank = [
                i,  # id
                self.params['initial_capital'],  # initial_capital
                self.params['max_bid'],  # max_bid
                self.params['initial_interest_rate_markup'],  # initial_interest_rate_markup
                self.params['banks_can_sell_bonds'][i],  # can_sell_bonds
                self.params['risk_appetite']  # risk_appetite
            ]
            banks.append(bank)
        return banks

    def load_consumers(self):
        consumers = []
        for i in range(self.params['num_consumers']):
            consumer = [
                i,  # id
                self.params['initial_capital'],  # initial_capital
                self.params['max_bid'],  # max_bid
                self.params['age'],  # age
                self.params['salary'],  # salary
                self.params['fixed_expenses'],  # fixed_expenses
                self.params['risk_appetite']  # risk_appetite
            ]
            consumers.append(consumer)
        return consumers

    def load_companies(self):
        companies = []
        for i in range(self.params['num_companies']):
            company = [
                i,  # id
                self.params['initial_capital'],  # initial_capital
                self.params['max_bid']  # max_bid
            ]
            companies.append(company)
        return companies

    def load_goods(self):
        goods = []
        for good in self.params['goods']:
            goods.append([
                self.category_to_id[good['category']],  # category ID
                good['weight'],  # weight
                good['initial_price']  # initial_price
            ])
        return goods