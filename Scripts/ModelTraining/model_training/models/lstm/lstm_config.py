import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
from .attention_layer import Attention, Attention2  # Adjust this import for relative path
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Register the custom layers for use in the model
tf.keras.utils.get_custom_objects().update({
    'Attention': Attention,
    'Attention2': Attention2
})

class LSTMModelConfig:
    @staticmethod
    def lstm_model(input_shape, params):
        model = Sequential()
        for layer in params['layers']:
            if layer['type'] == 'bidirectional_lstm':
                model.add(Bidirectional(LSTM(layer['units'], return_sequences=layer['return_sequences'], 
                                             input_shape=input_shape if not model.layers else None)))
            elif layer['type'] == 'attention':
                model.add(Attention())
            elif layer['type'] == 'attention2':
                model.add(Attention2())
            elif layer['type'] == 'batch_norm':
                model.add(BatchNormalization())
            elif layer['type'] == 'dropout':
                model.add(Dropout(layer['rate']))
            elif layer['type'] == 'dense':
                model.add(Dense(layer['units'], activation=layer['activation']))
        
        # Final output layer
        model.add(Dense(1))
        model.compile(optimizer=params['optimizer'], loss=params['loss'])
        return model


def main():
    data_file_path = 'C:/TheTradingRobotPlug/data/alpha_vantage/tsla_data.csv'
    model_save_path = 'best_model.keras'
    scaler_save_path = 'scaler.pkl'

    data_handler = DataHandler(logger=logger)
    data = data_handler.load_data(data_file_path)
    
    if data is not None:
        X_train, X_val, y_train, y_val = data_handler.preprocess_data(data)

        if X_train is not None and X_val is not None and y_train is not None and y_val is not None:
            time_steps = 10

            logger.info(f"X_train length: {len(X_train)}, y_train length: {len(y_train)}")
            logger.info(f"X_val length: {len(X_val)}, y_val length: {len(y_val)}")

            trainer = AdvancedLSTMModelTrainer(logger, model_save_path, scaler_save_path)
            try:
                logger.info(f"Creating sequences with time_steps: {time_steps}")
                X_train_seq, y_train_seq = trainer.create_sequences(X_train, y_train, time_steps)
                X_val_seq, y_val_seq = trainer.create_sequences(X_val, y_val, time_steps)

                logger.info(f"X_train_seq shape: {X_train_seq.shape}")
                logger.info(f"y_train_seq shape: {y_train_seq.shape}")
                logger.info(f"X_val_seq shape: {X_val_seq.shape}")
                logger.info(f"y_val_seq shape: {y_val_seq.shape}")

                model_params = {
                    'layers': [
                        {'type': 'bidirectional_lstm', 'units': 100, 'return_sequences': True, 'kernel_regularizer': None},
                        {'type': 'batch_norm'},
                        {'type': 'dropout', 'rate': 0.3},
                        {'type': 'dense', 'units': 50, 'activation': 'relu', 'kernel_regularizer': None}
                    ],
                    'optimizer': Adam(),  # Directly use the Adam optimizer object
                    'loss': 'mean_squared_error'
                }

                model_config = LSTMModelConfig.lstm_model((X_train_seq.shape[1], X_train_seq.shape[2]), model_params)
                
                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

                trained_model = trainer.train_lstm(
                    X_train_seq, y_train_seq, X_val_seq, y_val_seq, model_config, epochs=50,
                    callbacks=[early_stopping, reduce_lr]
                )

                if trained_model:
                    trainer.evaluate_model(X_val_seq, y_val_seq)
            except ValueError as e:
                logger.error(f"ValueError in creating sequences: {e}")
            except KeyError as e:
                logger.error(f"KeyError encountered: {e}")
        else:
            logger.error("Data preprocessing failed.")
    else:
        logger.error("Data loading failed.")

if __name__ == "__main__":
    main()
