import os
import sys
import tkinter as tk
from tkinter import ttk
import configparser
from tkinter.filedialog import askopenfilename
import keras

# Define the path you want to navigate to
path = r"C:\Users\Dagurlkc\OneDrive\Desktop\TradingRobotPlug"

# Change the current working directory to the specified path
os.chdir(path)

# Verify the current working directory
print("Current Working Directory: ", os.getcwd())

# Add the directory to the PYTHONPATH
sys.path.append(path)

# Debugging: Print the contents of the Tabs directory
tabs_path = os.path.join(path, 'Tabs')
print("Tabs Directory Content: ", os.listdir(tabs_path))

# Check if __init__.py exists in Tabs directory
if not os.path.isfile(os.path.join(tabs_path, '__init__.py')):
    with open(os.path.join(tabs_path, '__init__.py'), 'w') as f:
        pass  # Create an empty __init__.py file if it doesn't exist

# Check if __init__.py exists in risk_management_resources
risk_management_resources_path = os.path.join(tabs_path, 'risk_management_resources')
if not os.path.isfile(os.path.join(risk_management_resources_path, '__init__.py')):
    with open(os.path.join(risk_management_resources_path, '__init__.py'), 'w') as f:
        pass  # Create an empty __init__.py file if it doesn't exist

# Import custom modules
try:
    from Tabs.data_fetch_tab import DataFetchTab
    from Tabs.data_processing_tab import DataProcessingTab
    from Tabs.model_training_tab import ModelTrainingTab
    from Tabs.model_evaluation_tab import ModelEvaluationTab
    from Tabs.trade_analysis_tab import TradingAnalysisTab  
    from Tabs.trade_description_analyzer_tab import TradeDescriptionAnalyzerTab
    from Tabs.trade_analyzer_tab import TradeAnalyzerTab
    from Tabs.risk_management_tab import RiskManagementTab
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Global variable to store the trained model
trained_model = None

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')

# Create the main Tkinter window
root = tk.Tk()
root.title("Look...I did a thing")

# Create the tab control
tabControl = ttk.Notebook(root)

# Data Fetch Tab
data_fetch_tab = ttk.Frame(tabControl)
tabControl.add(data_fetch_tab, text='Data Fetch')
is_debug_mode = config.getboolean('Settings', 'DebugMode', fallback=False)
data_fetch_tab_instance = DataFetchTab(data_fetch_tab, config, is_debug_mode)

# Data Processing Tab
data_processing_tab = ttk.Frame(tabControl)
tabControl.add(data_processing_tab, text='Data Processing')
data_processing_tab_instance = DataProcessingTab(data_processing_tab, config)

# Model Training Tab
model_training_tab = ttk.Frame(tabControl)
tabControl.add(model_training_tab, text='Model Training')
scaler_options = ['standard', 'minmax', 'robust', 'normalizer', 'maxabs']

# Create an instance of ModelTrainingTab and pack it inside its parent frame
model_training_tab_instance = ModelTrainingTab(model_training_tab, config, scaler_options)
model_training_tab_instance.pack(fill="both", expand=True)

# Create the Risk Management Tab
risk_management_tab = ttk.Frame(tabControl)
tabControl.add(risk_management_tab, text='Risk Management')

# Assuming 'model_manager' is the instance of a class that should be passed to the RiskManagementTab
from Tabs.risk_management_resources.ModelManager import ModelManager
model_manager = ModelManager()
risk_management_tab_instance = RiskManagementTab(risk_management_tab, model_manager)

# Function to select a trained model
def select_trained_model():
    global trained_model
    model_path = askopenfilename(filetypes=[("Model Files", "*.h5")])
    if model_path:
        try:
            trained_model = keras.models.load_model(model_path)
            risk_management_tab_instance.set_trained_model(trained_model)
        except Exception as e:
            print(f"Error loading model: {e}")
            risk_management_tab_instance.set_trained_model(None)

# Bind the select trained model function to a button or menu item in the Risk Management Tab
button_select_model = ttk.Button(risk_management_tab_instance, text="Select Trained Model", command=select_trained_model)
button_select_model.pack()
risk_management_tab_instance.pack(fill="both", expand=True)

# Model Evaluation Tab
model_evaluation_tab = ttk.Frame(tabControl)
tabControl.add(model_evaluation_tab, text='Model Evaluation')

# Create an instance of ModelEvaluationTab and pass is_debug_mode
model_evaluation_tab_instance = ModelEvaluationTab(model_evaluation_tab, is_debug_mode)
model_evaluation_tab_instance.pack(fill="both", expand=True)

# Trade Analysis Tab
trade_analysis_tab = ttk.Frame(tabControl)
tabControl.add(trade_analysis_tab, text='Trade Analysis')
trade_analysis_tab_instance = TradingAnalysisTab(trade_analysis_tab)
trade_analysis_tab_instance.pack(fill="both", expand=True)

# Trade Description Analyzer Tab
trade_description_tab = ttk.Frame(tabControl)
tabControl.add(trade_description_tab, text='Trade Description Analyzer')
trade_description_tab_instance = TradeDescriptionAnalyzerTab(trade_description_tab)
trade_description_tab_instance.pack(fill="both", expand=True)

# Trade Analyzer Tab
trade_analyzer_tab = ttk.Frame(tabControl)
tabControl.add(trade_analyzer_tab, text='Trade Analyzer')
trade_analyzer_tab_instance = TradeAnalyzerTab(trade_analyzer_tab)
trade_analyzer_tab_instance.pack(fill="both", expand=True)

# Pack the tab control and run the main loop
tabControl.pack(expand=1, fill="both")

root.mainloop()
