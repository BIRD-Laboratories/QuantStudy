import json
import numpy as np
from sklearn.preprocessing import LabelEncoder

class DataLoader:
    def __init__(self, params_file):
        self.params_file = params_file
        self.params = self.load_parameters()
        self.category_to_id = {category: i for i, category in enumerate(self.params['goods_categories'])}
        self.label_encoders = {
            'family_status': LabelEncoder(),
            'spend_need': LabelEncoder(),
            'credit_status': LabelEncoder()
        }
        self.initialize_label_encoders()

    def load_parameters(self):
        with open(self.params_file, 'r') as file:
            params = json.load(file)
        return params

    def initialize_label_encoders(self):
        for key in self.label_encoders:
            values = [item['status' if key == 'family_status' or key == 'credit_status' else 'need'] for item in self.params[f'{key}_probabilities']]
            self.label_encoders[key].fit(values)

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
            age = np.random.choice([p['age'] for p in self.params['age_probabilities']], p=[p['probability'] for p in self.params['age_probabilities']])
            salary = np.random.choice([p['salary'] for p in self.params['salary_probabilities']], p=[p['probability'] for p in self.params['salary_probabilities']])
            family_status = self.label_encoders['family_status'].transform([np.random.choice([p['status'] for p in self.params['family_status_probabilities']], p=[p['probability'] for p in self.params['family_status_probabilities']])])[0]
            spend_need = self.label_encoders['spend_need'].transform([np.random.choice([p['need'] for p in self.params['spend_need_probabilities']], p=[p['probability'] for p in self.params['spend_need_probabilities']])])[0]
            credit_status = self.label_encoders['credit_status'].transform([np.random.choice([p['status'] for p in self.params['credit_status_probabilities']], p=[p['probability'] for p in self.params['credit_status_probabilities']])])[0]
            consumer = [
                i,  # id
                self.params['initial_capital'],  # initial_capital
                self.params['max_bid'],  # max_bid
                age,  # age
                salary,  # salary
                self.params['fixed_expenses'],  # fixed_expenses
                self.params['risk_appetite'],  # risk_appetite
                family_status,  # family_status
                spend_need,  # spend_need
                credit_status,  # credit_status
                0  # job_id (everyone has a job_id of 0 for now)
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