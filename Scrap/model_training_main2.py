import os
import logging
import robin_stocks.robinhood as r
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from a .env file (for security)
load_dotenv()

# Set up logging
logging.basicConfig(filename='trading_analysis.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Retrieve credentials securely from environment variables
username = os.getenv('ROBINHOOD_USERNAME')
password = os.getenv('ROBINHOOD_PASSWORD')

# Step 1: Log in to Robinhood with error handling
try:
    login = r.login(username, password)
    logging.info("Successfully logged into Robinhood.")
except Exception as e:
    logging.error(f"Failed to log in: {e}")
    raise SystemExit("Login failed, check credentials and 2FA settings.")

# Step 2: Ask the user whether to analyze at the individual contract level or trade level
analysis_type = input("Do you want to analyze at the (1) individual contract level or (2) trade level? Enter 1 or 2: ")

# Step 3: Retrieve Options Orders with error handling
try:
    options_orders = r.options.get_all_option_orders()
    logging.info("Successfully retrieved options orders.")
except Exception as e:
    logging.error(f"Failed to retrieve options orders: {e}")
    raise SystemExit("Failed to retrieve options orders.")

# Step 4: Filter and Organize the Data based on User Choice
if analysis_type == '1':
    # Individual Contract Analysis
    contracts = []
    try:
        for order in options_orders:
            if order['state'] == 'filled':
                for leg in order['legs']:
                    contract_quantity = int(float(leg['executed_quantity']))
                    contract_price = float(leg['price'])

                    for _ in range(contract_quantity):
                        profit_loss = (float(order['price']) - contract_price)

                        contract = {
                            'Date': order['created_at'],
                            'Option': leg['option_id'],  # or leg['option']
                            'Type': leg['side'],
                            'Price': contract_price,
                            'Profit/Loss': profit_loss,
                            'State': order['state']
                        }
                        contracts.append(contract)
        logging.info("Successfully processed individual option contracts.")
    except Exception as e:
        logging.error(f"Failed to process contracts: {e}")
        raise SystemExit("Failed to process contracts.")

    # Convert to DataFrame
    analysis_df = pd.DataFrame(contracts)
else:
    # Trade-Level Analysis
    trades = []
    try:
        for order in options_orders:
            if order['state'] == 'filled':
                for leg in order['legs']:
                    profit_loss = float(order['cumulative_quantity']) * (float(order['price']) - float(leg['price']))

                    trade = {
                        'Date': order['created_at'],
                        'Option': leg['option_id'],  # or leg['option']
                        'Type': leg['side'],
                        'Quantity': leg['executed_quantity'],
                        'Price': leg['price'],
                        'Profit/Loss': profit_loss,
                        'State': order['state']
                    }
                    trades.append(trade)
        logging.info("Successfully processed option trades.")
    except Exception as e:
        logging.error(f"Failed to process trades: {e}")
        raise SystemExit("Failed to process trades.")

    # Convert to DataFrame
    analysis_df = pd.DataFrame(trades)

# Step 5: Calculate Accuracy and Other Metrics
try:
    winning_trades = analysis_df[analysis_df['Profit/Loss'] > 0].shape[0]
    losing_trades = analysis_df[analysis_df['Profit/Loss'] <= 0].shape[0]
    total_trades = winning_trades + losing_trades
    accuracy = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

    average_profit = analysis_df[analysis_df['Profit/Loss'] > 0]['Profit/Loss'].mean()
    average_loss = analysis_df[analysis_df['Profit/Loss'] <= 0]['Profit/Loss'].mean()

    total_profit = analysis_df[analysis_df['Profit/Loss'] > 0]['Profit/Loss'].sum()
    total_loss = analysis_df[analysis_df['Profit/Loss'] <= 0]['Profit/Loss'].sum()

    # Display the results
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {winning_trades}")
    print(f"Losing Trades: {losing_trades}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average Profit: ${average_profit:.2f}")
    print(f"Average Loss: ${average_loss:.2f}")
    print(f"Total Profit: ${total_profit:.2f}")
    print(f"Total Loss: ${total_loss:.2f}")

    logging.info(f"Trading analysis completed successfully.")
except Exception as e:
    logging.error(f"Failed to calculate metrics: {e}")
    raise SystemExit("Failed to calculate metrics.")

# Optional: Save the data to a CSV for further analysis
output_filename = 'robinhood_contracts_history.csv' if analysis_type == '1' else 'robinhood_trades_history.csv'
try:
    analysis_df.to_csv(output_filename, index=False)
    logging.info(f"Successfully saved trading history to {output_filename}.")
except Exception as e:
    logging.error(f"Failed to save trading history to CSV: {e}")
    raise SystemExit("Failed to save trading history.")
