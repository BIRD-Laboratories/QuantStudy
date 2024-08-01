import json

def load_parameters(file_path):
    with open(file_path, 'r') as file:
        params = json.load(file)
    return params