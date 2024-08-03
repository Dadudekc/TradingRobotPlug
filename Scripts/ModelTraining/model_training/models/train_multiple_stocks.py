import yaml
from models.arima_model_trainer import ARIMAModelTrainer
from pathlib import Path

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    config_file_path = project_root / 'config.yaml'
    config = load_config(config_file_path)
    
    for stock in config['stocks']:
        symbol = stock['symbol']
        trainer = ARIMAModelTrainer(symbol=symbol, threshold=stock.get('threshold', 100))
        trainer.train()

if __name__ == '__main__':
    main()
