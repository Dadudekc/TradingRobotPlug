from Scripts.DataFetchers.base_fetcher import DataFetcher
from Scripts.DataFetchers.real_time_fetcher import RealTimeDataFetcher

class PolygonDataFetcher(DataFetcher, RealTimeDataFetcher):
    def __init__(self):
        super().__init__('POLYGON_API_KEY', 'https://api.polygon.io/v2/aggs/ticker', 'CSV_DIR', 'DB_PATH', 'POLYGON_LOG_FILE', 'Polygon')

    def construct_api_url(self, symbol: str, start_date: str, end_date: str) -> str:
        return f"{self.base_url}/{symbol}/range/1/day/{start_date}/{end_date}?apiKey={self.api_key}"

    def extract_results(self, data: dict) -> list:
        results = data.get('results', [])
        self.utils.logger.debug(f"Polygon: Extracted results: {results}")
        return [
            {
                'date': datetime.utcfromtimestamp(result['t'] / 1000).strftime('%Y-%m-%d'),
                'open': result['o'],
                'high': result['h'],
                'low': result['l'],
                'close': result['c'],
                'volume': result['v']
            }
            for result in results
        ]

    def construct_real_time_api_url(self, symbol: str) -> str:
        return f"https://api.polygon.io/v1/last/stocks/{symbol}?apiKey={self.api_key}"

    def extract_real_time_results(self, data: dict) -> list:
        result = data.get('results', {})
        return [
            {
                'timestamp': datetime.utcfromtimestamp(result['t'] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                'price': result['price'],
                'size': result['size']
            }
        ]
