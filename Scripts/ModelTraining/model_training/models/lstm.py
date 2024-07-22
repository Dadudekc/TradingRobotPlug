import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score

def train_lstm(X_train, y_train, X_val, y_val, model_config, epochs=100, pretrained_model_path=None, logger=None):
    """Train an LSTM model."""
    if pretrained_model_path:
        model = load_model(pretrained_model_path)
        for layer in model.layers[:-5]:
            layer.trainable = False
    else:
        model = Sequential()

    model.add(Input(shape=(X_train.shape[1], 1)))
    for layer in model_config['layers']:
        if layer['type'] == 'lstm':
            model.add(LSTM(units=layer['units'], return_sequences=layer['return_sequences'], kernel_regularizer=layer['kernel_regularizer']))
        elif layer['type'] == 'batch_norm':
            model.add(BatchNormalization())
        elif layer['type'] == 'dropout':
            model.add(Dropout(rate=layer['rate']))
        elif layer['type'] == 'dense':
            model.add(Dense(units=layer['units'], activation=layer['activation'], kernel_regularizer=layer['kernel_regularizer']))

    model.compile(optimizer=model_config['optimizer'], loss=model_config['loss'])

    X_train_reshaped = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val_reshaped = X_val.values.reshape(X_val.shape[0], X_val.shape[1], 1)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train_reshaped, y_train, validation_data=(X_val_reshaped, y_val), epochs=epochs, batch_size=32, callbacks=[early_stopping])

    y_pred_val = model.predict(X_val_reshaped).flatten()
    mse = mean_squared_error(y_val, y_pred_val)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, y_pred_val)

    if logger:
        logger.info(f"Validation MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

    return model
