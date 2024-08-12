import numpy as np

class VIX:
    def __init__(self):
        self.vix_history = []

    def calculate_vix(self, volatility):
        """
        Calculate the VIX based on the given volatility.
        For simplicity, we'll assume volatility is directly proportional to VIX.
        """
        vix = volatility * 10  # Example calculation, adjust as needed
        self.vix_history.append(vix)
        return vix

    def get_vix_history(self):
        return self.vix_history