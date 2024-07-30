from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, BatchNormalization, Dropout, Dense
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Layer
import tensorflow as tf

class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1]), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[-1],), initializer="zeros", trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

class LSTMModelConfig:
    @staticmethod
    def lstm_model(input_shape, model_params):
        model = Sequential()
        
        if not model_params['layers']:
            raise ValueError("Model configuration should include at least one layer.")

        for layer in model_params['layers']:
            if layer['type'] == 'bidirectional_lstm':
                if 'input_shape' in layer:
                    model.add(Bidirectional(LSTM(units=layer['units'], return_sequences=layer['return_sequences'], kernel_regularizer=layer['kernel_regularizer']),
                                            input_shape=input_shape))
                else:
                    model.add(Bidirectional(LSTM(units=layer['units'], return_sequences=layer['return_sequences'], kernel_regularizer=layer['kernel_regularizer'])))
            elif layer['type'] == 'batch_norm':
                model.add(BatchNormalization())
            elif layer['type'] == 'dropout':
                model.add(Dropout(rate=layer['rate']))
            elif layer['type'] == 'dense':
                model.add(Dense(units=layer['units'], activation=layer['activation'], kernel_regularizer=layer['kernel_regularizer']))
            elif layer['type'] == 'attention':
                model.add(Attention())
            else:
                raise ValueError(f"Unsupported layer type: {layer['type']}")
        
        model.add(Dense(1))  # Ensure the output layer has a single unit
        
        model.compile(optimizer=model_params['optimizer'], loss='mean_squared_error')
        return model
