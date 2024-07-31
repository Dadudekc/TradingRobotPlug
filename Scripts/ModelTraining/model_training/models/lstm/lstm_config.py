from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
from .attention_layer import Attention, Attention2  # Adjust this import for relative path
import tensorflow as tf

# Register the custom layers
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
                model.add(Bidirectional(LSTM(layer['units'], return_sequences=layer['return_sequences'], input_shape=input_shape if model.layers == [] else None)))
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
        # Ensure the final layer produces a single output per sequence
        model.add(Dense(1))
        model.compile(optimizer=params['optimizer'], loss=params['loss'])
        return model
