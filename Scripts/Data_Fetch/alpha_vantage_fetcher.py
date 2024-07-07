from Scripts.DataFetchers.base_fetcher import DataFetcher
from Scripts.DataFetchers.real_time_fetcher import RealTimeDataFetcher

class AlphaVantageDataFetcher(DataFetcher, RealTimeDataFetcher):
    def __init__(self):
        super().__init__('ALPHAVANTAGE_API_KEY', 'https://www.alphavantage.co/query', 'CSV_DIR', 'DB_PATH', 'ALPHA_LOG_FILE', 'AlphaVantage')

    def construct_api_url(self, symbol: str, start_date: str, end_date: str) -> str:
        return f"{self.base_url}?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={self.api_key}&outputsize=full&datatype=json"

    def extract_results(self, data: dict) -> list:
        time_series = data.get("Time Series (Daily)", {})
        results = [
            {
                'date': date,
                'open': float(values["1. open"]),
                'high': float(values["2. high"]),
                'low': float(values["3. low"]),
                'close': float(values["4. close"]),
                'volume': int(values["5. volume"])
            }
            for date, values in time_series.items()
        ]
        self.utils.logger.debug(f"AlphaVantage: Extracted results: {results}")
        return results

    def construct_real_time_api_url(self, symbol: str) -> str:
        return f"{self.base_url}?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=1min&apikey={self.api_key}&datatype=json"

    def extract_real_time_results(self, data: dict) -> list:
        time_series = data.get("Time Series (1min)", {})
        results = [
            {
                'timestamp': timestamp,
                'open': float(values["1. open"]),
                'high': float(values["2. high"]),
                'low': float(values["3. low"]),
                'close': float(values["4. close"]),
                'volume': int(values["5. volume"])
            }
            for timestamp, values in time_series.items()
        ]
        self.utils.logger.debug(f"AlphaVantage: Extracted real-time results: {results}")
        return results
