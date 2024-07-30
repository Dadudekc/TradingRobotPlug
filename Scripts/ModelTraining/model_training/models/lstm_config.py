class LSTMModelConfig:
    @staticmethod
    def lstm_model(input_shape, params):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional

        model = Sequential()
        for layer in params['layers']:
            if layer['type'] == 'bidirectional_lstm':
                model.add(Bidirectional(LSTM(layer['units'], return_sequences=layer['return_sequences'])))
            elif layer['type'] == 'attention':
                model.add(Attention())
            elif layer['type'] == 'batch_norm':
                model.add(BatchNormalization())
            elif layer['type'] == 'dropout':
                model.add(Dropout(layer['rate']))
            elif layer['type'] == 'dense':
                model.add(Dense(layer['units'], activation=layer['activation']))
        model.compile(optimizer=params['optimizer'], loss=params['loss'])
        return model

# Ensure you have the necessary imports and the Attention layer as before.
