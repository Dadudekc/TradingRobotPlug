from Scripts.DataFetchers.base_fetcher import DataFetcher
from Scripts.DataFetchers.real_time_fetcher import RealTimeDataFetcher

class NasdaqDataFetcher(DataFetcher, RealTimeDataFetcher):
    def __init__(self):
        super().__init__('NASDAQ_API_KEY', 'https://dataondemand.nasdaq.com/api/v1/historical', 'CSV_DIR', 'DB_PATH', 'NASDAQ_LOG_FILE', 'Nasdaq')

    def construct_api_url(self, symbol: str, start_date: str, end_date: str) -> str:
        url = f"{self.base_url}/{symbol}?apiKey={self.api_key}"
        if start_date:
            url += f"&startDate={start_date}"
        if end_date:
            url += f"&endDate={end_date}"
        return url

    def extract_results(self, data: dict) -> list:
        results = data.get('data', [])
        self.utils.logger.debug(f"Nasdaq: Extracted results: {results}")
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
        return f"{self.base_url}/real-time/{symbol}?apiKey={self.api_key}"

    def extract_real_time_results(self, data: dict) -> list:
        results = data.get('data', [])
        return [
            {
                'timestamp': datetime.utcfromtimestamp(result['t'] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                'open': result['o'],
                'high': result['h'],
                'low': result['l'],
                'close': result['c'],
                'volume': result['v']
            }
            for result in results
        ]
