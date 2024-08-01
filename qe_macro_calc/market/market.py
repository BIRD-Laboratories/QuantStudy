class Market:
    def __init__(self, liquidity_factor):
        self.liquidity_factor = liquidity_factor

    def impact_asset_prices(self, assets, interest_rate):
        for asset in assets:
            asset.price *= (1 + interest_rate * self.liquidity_factor)
