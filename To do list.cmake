To do list

setup other computer to work on files


debug tests

C:\TheTradingRobotPlug\Tests\Utilities

C:\TheTradingRobotPlug\Tests\Utilities\test_config_handling.py
C:\TheTradingRobotPlug\Tests\Utilities\test_data_fetch_utils.py
C:\TheTradingRobotPlug\Tests\Utilities\test_data_processing_utils.py
C:\TheTradingRobotPlug\Tests\Utilities\test_data_store.py
C:\TheTradingRobotPlug\Tests\Utilities\test_DataLakeHandler.py

create tests 

C:\TheTradingRobotPlug\Model_Training
C:\TheTradingRobotPlug\Model_Training\__pycache__
C:\TheTradingRobotPlug\Model_Training\__init__.py
C:\TheTradingRobotPlug\Model_Training\AutomatedModelTrainer.py
C:\TheTradingRobotPlug\Model_Training\DataHandler.py
C:\TheTradingRobotPlug\Model_Training\HyperparameterTuner.py
C:\TheTradingRobotPlug\Model_Training\ModelEvaluator.py
C:\TheTradingRobotPlug\Model_Training\ModelTrainer.py

C:\TheTradingRobotPlug\ModelTraining
C:\TheTradingRobotPlug\ModelTraining\__pycache__
C:\TheTradingRobotPlug\ModelTraining\backtest.py
C:\TheTradingRobotPlug\ModelTraining\continuous_learning.py
C:\TheTradingRobotPlug\ModelTraining\data_preprocessing.py
C:\TheTradingRobotPlug\ModelTraining\gui_module.py
C:\TheTradingRobotPlug\ModelTraining\hyperparameter_tuning.py
C:\TheTradingRobotPlug\ModelTraining\logging_module.py
C:\TheTradingRobotPlug\ModelTraining\Model_training_tab_main.py
C:\TheTradingRobotPlug\ModelTraining\model_training.py
C:\TheTradingRobotPlug\ModelTraining\trading_env.py
C:\TheTradingRobotPlug\ModelTraining\train_drl_model.py
C:\TheTradingRobotPlug\ModelTraining\utilities.py
-------------------------------------------

integrate

C:\TheTradingRobotPlug\Tests
C:\TheTradingRobotPlug\Tests\__pycache__
C:\TheTradingRobotPlug\Tests\data\csv
C:\TheTradingRobotPlug\Tests\Data_Fetch
C:\TheTradingRobotPlug\Tests\Data_Fetch\__pycache__
C:\TheTradingRobotPlug\Tests\Data_Fetch\__init__.py
C:\TheTradingRobotPlug\Tests\Data_Fetch\test_alpha_vantage_fetcher.py
C:\TheTradingRobotPlug\Tests\Data_Fetch\test_api_interaction.py
C:\TheTradingRobotPlug\Tests\Data_Fetch\test_base_fetcher.py
C:\TheTradingRobotPlug\Tests\Data_Fetch\test_data_fetcher.py
C:\TheTradingRobotPlug\Tests\Data_Fetch\test_polygon_fetcher.py
C:\TheTradingRobotPlug\Tests\Data_Fetch\test_real_time_fetcher.py
C:\TheTradingRobotPlug\Tests\Data_Fetch\test.py
C:\TheTradingRobotPlug\Tests\Data_Processing\Technical_indicators
C:\TheTradingRobotPlug\Tests\Data_Processing\Technical_indicators\test_custom_indicators.py
C:\TheTradingRobotPlug\Tests\Data_Processing\Technical_indicators\test_momentum_indicators.py
C:\TheTradingRobotPlug\Tests\Data_Processing\Technical_indicators\test_trend_indicators.py
C:\TheTradingRobotPlug\Tests\Data_Processing\Technical_indicators\test_volatility_indicators.py
C:\TheTradingRobotPlug\Tests\Data_Processing\Technical_indicators\test_volume_indicators.py
C:\TheTradingRobotPlug\Tests\GUI
C:\TheTradingRobotPlug\Tests\GUI\test_base_gui.py
C:\TheTradingRobotPlug\Tests\GUI\test_data_fetch_tab.py
C:\TheTradingRobotPlug\Tests\GUI\test_fetcher_gui.py
C:\TheTradingRobotPlug\Tests\logs
C:\TheTradingRobotPlug\Tests\logs\data_fetch_utils.log
C:\TheTradingRobotPlug\Tests\mock_csv_dir
C:\TheTradingRobotPlug\Tests\test_csv_dir
C:\TheTradingRobotPlug\Tests\test_csv_dir\processed
C:\TheTradingRobotPlug\Tests\test_csv_dir\raw
C:\TheTradingRobotPlug\Tests\test_log_dir
C:\TheTradingRobotPlug\Tests\test_log_dir\test_log_file.log
C:\TheTradingRobotPlug\Tests\Utilities
C:\TheTradingRobotPlug\Tests\Utilities\test_config_handling.py
C:\TheTradingRobotPlug\Tests\Utilities\test_data_fetch_utils.py
C:\TheTradingRobotPlug\Tests\Utilities\test_data_processing_utils.py
C:\TheTradingRobotPlug\Tests\Utilities\test_data_store.py
C:\TheTradingRobotPlug\Tests\Utilities\test_DataLakeHandler.py

together into one cohesive unit here:

C:\TheTradingRobotPlug\Tests\run_tests.py


----------------------------------------





----------------------------------------


# Example configuration
.env file = {}  # load the environment
log_text_widget = None  # Your Tkinter Text widget for logging (optional)

# Initialize DataHandler and load data
data_handler = DataHandler(config, log_text_widget)
data = data_handler.load_data('data/file/path.csv')

# Preprocess data
X, y = data_handler.preprocess_data(data)
X_train, X_val, y_train, y_val = data_handler.split_data(X, y)
X_train_scaled, X_val_scaled, scaler = data_handler.scale_data(X_train, X_val)

# Initialize HyperparameterTuner
tuner = HyperparameterTuner(logger=ModelTrainingLogger(log_text_widget))

# Initialize and configure model
model = tuner.initialize_and_configure_model('neural_network', (X_train_scaled.shape[1],), epochs=50)

# Perform hyperparameter tuning
param_distributions = {
    'layers': [1, 2, 3],
    'units': [32, 64, 128],
    'dropout': [0.1, 0.2, 0.3]
}
best_model, best_params = tuner.perform_hyperparameter_tuning(model, X_train_scaled, y_train, param_distributions)

# Initialize ModelTrainer and train the model
trainer = ModelTrainer(config, log_text_widget)
trainer.trained_model = best_model  # Assuming the best model is returned from the tuner
trainer.save_trained_model(best_model, 'neural_network', scaler)

# Initialize ModelEvaluator and evaluate the model
evaluator = ModelEvaluator(log_text_widget)
evaluation_results = evaluator.evaluate_model(best_model, X_val_scaled, y_val, 'regression')
evaluator.visualize_training_results(y_val, best_model.predict(X_val_scaled))
evaluator.generate_model_reports(best_model, X_val_scaled, y_val, best_model.predict(X_val_scaled), 'regression')
