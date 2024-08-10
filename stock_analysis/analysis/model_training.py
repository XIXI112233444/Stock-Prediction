import logging
import os
import pandas as pd
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from keras.models import Sequential
from keras.layers import GRU, LSTM, Dense, Dropout, InputLayer
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau
from keras_tuner import BayesianOptimization
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm.keras import TqdmCallback
from keras.optimizers import Adam
from scipy.stats import uniform, loguniform
import pickle
from sklearn.linear_model import LinearRegression

# Custom dynamic early stopping callback
class DynamicEarlyStopping(Callback):
    def __init__(self, patience=5, initial_threshold=100000, min_threshold=10000, threshold_decay=0.9):
        super(DynamicEarlyStopping, self).__init__()
        self.patience = patience
        self.initial_threshold = initial_threshold
        self.min_threshold = min_threshold
        self.threshold_decay = threshold_decay
        self.best_weights = None
        self.bad_epochs = 0
        self.best_val_mse = float('inf')
        self.current_threshold = initial_threshold

    def on_epoch_end(self, epoch, logs=None):
        val_mse = logs.get("val_mse")
        if val_mse is None:
            return

        if (val_mse > self.current_threshold):
            self.bad_epochs += 1
            logging.debug(f"Bad epoch {self.bad_epochs}: val_mse {val_mse} exceeded threshold {self.current_threshold}")
        else:
            self.bad_epochs = 0
            self.best_val_mse = min(self.best_val_mse, val_mse)
            self.current_threshold = max(self.min_threshold, self.current_threshold * self.threshold_decay)
            logging.debug(f"Epoch {epoch}: val_mse {val_mse} improved, updating threshold to {self.current_threshold}")
            self.best_weights = self.model.get_weights()

        if self.bad_epochs >= self.patience:
            logging.debug(f"Stopping training at epoch {epoch} due to no improvement in {self.patience} consecutive epochs.")
            self.model.stop_training = True
            if self.best_weights is not None:
                self.model.set_weights(self.best_weights)

# Custom callback for additional metrics
class CustomMetrics(Callback):
    def __init__(self, validation_data, target_scaler, model_type, best_params):
        super().__init__()
        self.validation_data = validation_data
        self.target_scaler = target_scaler
        self.model_type = model_type
        self.best_val_mse = float('inf')
        self.best_params = best_params

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        y_val_pred = self.model.predict(X_val)
        y_val_true = y_val

        y_val_pred_2d = y_val_pred.reshape(-1, 1)
        y_val_true_2d = y_val.reshape(-1, 1)

        y_val_pred_rescaled = self.target_scaler.inverse_transform(y_val_pred_2d)
        y_val_true_rescaled = self.target_scaler.inverse_transform(y_val_true_2d)

        y_val_pred_rescaled = y_val_pred_rescaled.reshape(y_val_pred.shape)
        y_val_true_rescaled = y_val_true_rescaled.reshape(y_val.shape)

        y_val_true_flat = y_val_true_rescaled.flatten()
        y_val_pred_flat = y_val_pred_rescaled.flatten()

        min_len = min(len(y_val_true_flat), len(y_val_pred_flat))
        y_val_true_flat = y_val_true_flat[:min_len]
        y_val_pred_flat = y_val_pred_flat[:min_len]

        mse = mean_squared_error(y_val_true_flat, y_val_pred_flat)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val_true_flat, y_val_pred_flat)
        r2 = r2_score(y_val_true_flat, y_val_pred_flat)
        mape = np.mean(np.abs((y_val_true_flat - y_val_pred_flat) / np.where(y_val_true_flat == 0, 1e-10, y_val_true_flat))) * 100
        max_ape = np.max(np.abs((y_val_true_flat - y_val_pred_flat) / np.where(y_val_true_flat == 0, 1e-10, y_val_true_flat))) * 100
        direction_accuracy = np.mean((np.sign(y_val_true_flat[1:] - y_val_true_flat[:-1]) == np.sign(y_val_pred_flat[1:] - y_val_pred_flat[:-1])).astype(int))

        logs['val_mse'] = mse
        logs['scaled_val_mse'] = mse
        logs['val_rmse'] = rmse
        logs['val_mae'] = mae
        logs['val_r2'] = r2
        logs['val_mape'] = mape
        logs['val_max_ape'] = max_ape
        logs['val_direction_accuracy'] = direction_accuracy
        print(f" — val_mse: {mse:.4f} — val_rmse: {rmse:.4f} — val_mae: {mae:.4f} — val_r2: {r2:.4f} — val_mape: {mape:.4f}% — val_max_ape: {max_ape:.4f}% — val_direction_accuracy: {direction_accuracy:.4f}")

        # Debugging output to ensure val_mse is being set correctly
        print(f"[DEBUG] CustomMetrics on_epoch_end: val_mse = {mse}")
        print(f"[DEBUG] CustomMetrics on_epoch_end: scaled_val_mse = {logs['scaled_val_mse']}")

        # Save best parameters if the current model is better
        best_params_path = f'best_{self.model_type}_params.pkl'
        if not os.path.exists(best_params_path) or mse < self.best_val_mse:
            self.best_val_mse = mse
            self.best_params['performance'] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape,
                'max_ape': max_ape,
                'direction_accuracy': direction_accuracy
            }
            with open(best_params_path, 'wb') as f:
                pickle.dump(self.best_params, f)
            self.model.save(f'best_{self.model_type}_model.keras')

# Data preparation function
def prepare_data(data, time_steps):
    features = data.drop(columns=['Date', 'Window', 'Close', 'Adj Close'])
    target = data['Close']
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    features_scaled = feature_scaler.fit_transform(features)
    target_scaled = target_scaler.fit_transform(target.values.reshape(-1, 1))
    X, y = [], []
    for i in range(time_steps, len(features_scaled)):
        X.append(features_scaled[i-time_steps:i])
        y.append(target_scaled[i])
    X, y = np.array(X), np.array(y)
    return X, y, feature_scaler, target_scaler

# Hyperparameter model building function
def build_hyperparameter_model(best_params, input_shape, model_type='gru'):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    if (model_type == 'gru'):
        for i in range(3):  # 3-layer GRU
            units = best_params['units'][i]
            dropout_rate = best_params['dropout_rate'][i]
            model.add(GRU(units=units, activation='tanh', recurrent_activation='sigmoid', use_bias=True,
                          kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                          bias_initializer='zeros', dropout=dropout_rate, recurrent_dropout=0.0, 
                          return_sequences=True if i < 2 else False))
            model.add(Dropout(rate=dropout_rate))
    elif model_type == 'lstm':
        for i in range(3):  # 3-layer LSTM
            units = best_params['units'][i]
            dropout_rate = best_params['dropout_rate'][i]
            model.add(LSTM(units=units, return_sequences=True if i < 2 else False, 
                           activation='tanh', kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', 
                           bias_initializer='zeros', dropout=dropout_rate))
            model.add(Dropout(rate=dropout_rate))
    elif model_type == '12gru':
        for i in range(12):  # 12-layer GRU
            units = best_params['units'][i]
            dropout_rate = best_params['dropout_rate'][i]
            model.add(GRU(units=units, activation='tanh', recurrent_activation='sigmoid', use_bias=True,
                          kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                          bias_initializer='zeros', dropout=dropout_rate, recurrent_dropout=0.0,
                          return_sequences=True if i < 11 else False))
            model.add(Dropout(rate=dropout_rate))
    model.add(Dense(1, activation='linear', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    return model

# 12-layer GRU model building function
def build_12gru_model(input_shape, units, dropout_rate):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    for i in range(12):
        model.add(GRU(units=units[i], activation='tanh', recurrent_activation='sigmoid', use_bias=True,
                      kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                      bias_initializer='zeros', dropout=dropout_rate[i], recurrent_dropout=0.0,
                      return_sequences=True if i < 11 else False))
        model.add(Dropout(rate=dropout_rate[i]))
    model.add(Dense(1, activation='linear', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    return model

# Final model training with hyperparameter tuning function
def train_final_model(X_train, y_train, X_val, y_val, input_shape, target_scaler, epochs=100, batch_size=32, model_type='gru', additional_callbacks=[]):
    best_params_path = f'best_{model_type}_params.pkl'

    if model_type == '12gru':
        best_params = None
        if os.path.exists(best_params_path):
            with open(best_params_path, 'rb') as f:
                best_params = pickle.load(f)

        if best_params:
            best_model = build_12gru_model(input_shape, best_params['units'], best_params['dropout_rate'])
        else:
            tuner = BayesianOptimization(
                lambda hp: build_hyperparameter_model({
                    'units': [hp.Int(f'units_{i+1}', min_value=64, max_value=320, step=64) for i in range(12)],
                    'dropout_rate': [hp.Float(f'dropout_rate_{i+1}', min_value=0.0, max_value=0.5, step=0.1) for i in range(12)]
                }, input_shape, model_type='12gru'),
                objective='val_mse',
                max_trials=100,  # Set to a reasonable value to control hyperparameter optimization times
                executions_per_trial=1,
                directory='hyperparameter_tuning',
                project_name=f'{model_type}_model'
            )
            tuner.search_space_summary()
            custom_metrics = CustomMetrics((X_val, y_val), target_scaler, model_type, best_params={})  # Add model_type parameter
            early_stopping = DynamicEarlyStopping(patience=5, initial_threshold=100000, min_threshold=10000, threshold_decay=0.9)
            tuner.search(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), callbacks=[custom_metrics, early_stopping] + additional_callbacks)

            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            best_params = {
                'units': [best_hps.get(f'units_{i+1}') for i in range(12)],
                'dropout_rate': [best_hps.get(f'dropout_rate_{i+1}') for i in range(12)]
            }

            best_model = tuner.hypermodel.build(best_hps)
            
            # Save the best parameters
            with open(best_params_path, 'wb') as f:
                pickle.dump(best_params, f)
    else:
        best_params = None
        if os.path.exists(best_params_path):
            with open(best_params_path, 'rb') as f:
                best_params = pickle.load(f)

        if best_params:
            best_model = build_hyperparameter_model(best_params, input_shape, model_type=model_type)
        else:
            tuner = BayesianOptimization(
                lambda hp: build_hyperparameter_model({
                    'units': [hp.Int(f'units_{i+1}', min_value=64, max_value=320, step=64) for i in range(3)],
                    'dropout_rate': [hp.Float(f'dropout_rate_{i+1}', min_value=0.0, max_value=0.5, step=0.1) for i in range(3)]
                }, input_shape, model_type=model_type),
                objective='val_mse',
                max_trials=100,  # Set to a reasonable value to control hyperparameter optimization times
                executions_per_trial=1,
                directory='hyperparameter_tuning',
                project_name=f'{model_type}_model'
            )
            tuner.search_space_summary()
            custom_metrics = CustomMetrics((X_val, y_val), target_scaler, model_type, best_params={})  # Add model_type parameter
            early_stopping = DynamicEarlyStopping(patience=5, initial_threshold=100000, min_threshold=10000, threshold_decay=0.9)
            tuner.search(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), callbacks=[custom_metrics, early_stopping] + additional_callbacks)

            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            best_params = {
                'units': [best_hps.get(f'units_{i+1}') for i in range(3)],
                'dropout_rate': [best_hps.get(f'dropout_rate_{i+1}') for i in range(3)]
            }

            best_model = tuner.hypermodel.build(best_hps)
            
            # Save the best parameters
            with open(best_params_path, 'wb') as f:
                pickle.dump(best_params, f)

    checkpoint = ModelCheckpoint(f'best_{model_type}_model.keras', monitor='val_loss', save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    custom_metrics = CustomMetrics((X_val, y_val), target_scaler, model_type, best_params)  # Add model_type and best_params parameters
    early_stopping = DynamicEarlyStopping(patience=5, initial_threshold=100000, min_threshold=10000, threshold_decay=0.9)
    history = best_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[custom_metrics, early_stopping, checkpoint, TqdmCallback(verbose=1), reduce_lr] + additional_callbacks)
    
    return best_model, history

# SVM training with hyperparameter optimization function
def train_svm_with_hyperparameter_optimization(X_train, y_train, X_val, y_val, target_scaler, epochs=100):
    param_distributions = {
        'C': loguniform(0.1, 100),
        'epsilon': uniform(0.01, 0.5)
    }

    svr = SVR(kernel='rbf')

    # Ensure y_train and y_val are 1D
    y_train = y_train.ravel()
    y_val = y_val.ravel()

    random_search = RandomizedSearchCV(svr, param_distributions, n_iter=100, scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=1)
    random_search.fit(X_train.reshape(X_train.shape[0], -1), y_train)

    best_model = random_search.best_estimator_

    print(f"""
    The hyperparameter search for SVM is complete. The optimal value for C is {best_model.C}
    the optimal value for epsilon is {best_model.epsilon}
    """)

    best_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)

    history = []
    for epoch in range(epochs):
        best_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        y_val_pred = best_model.predict(X_val.reshape(X_val.shape[0], -1))
        mse = mean_squared_error(y_val, y_val_pred)
        history.append({'epoch': epoch, 'val_mse': mse})
        
        # Save the best parameters if current model is better
        if epoch == 0 or mse < min([h['val_mse'] for h in history]):
            best_params = {
                'C': best_model.C,
                'epsilon': best_model.epsilon,
                'performance': {
                    'mse': mse,
                    'rmse': np.sqrt(mse),
                    'mae': mean_absolute_error(y_val, y_val_pred),
                    'r2': r2_score(y_val, y_val_pred),
                    'mape': np.mean(np.abs((y_val - y_val_pred) / np.where(y_val == 0, 1e-10, y_val))) * 100,
                    'direction_accuracy': np.mean((np.sign(y_val[1:] - y_val[:-1]) == np.sign(y_val_pred[1:] - y_val_pred[:-1])).astype(int)),
                    'max_ape': np.max(np.abs((y_val - y_val_pred) / np.where(y_val == 0, 1e-10, y_val))) * 100
                }
            }
            with open('best_svm_params.pkl', 'wb') as f:
                pickle.dump(best_params, f)
            pickle.dump(best_model, open('best_svm_model.pkl', 'wb'))
    
    return best_model, history

# Train linear regression model with hyperparameter search
def train_linear_regression_with_hyperparameter_search(X_train, y_train, X_val, y_val, target_scaler, epochs=100):
    linear_model = LinearRegression()
    # Ensure y_train and y_val are 1D
    y_train = y_train.ravel()
    y_val = y_val.ravel()

    history = []
    for epoch in range(epochs):
        linear_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        y_val_pred = linear_model.predict(X_val.reshape(X_val.shape[0], -1))
        mse = mean_squared_error(y_val, y_val_pred)
        history.append({'epoch': epoch, 'val_mse': mse})

        # Save the best parameters if current model is better
        if epoch == 0 or mse < min([h['val_mse'] for h in history]):
            best_params = {
                'performance': {
                    'mse': mse,
                    'rmse': np.sqrt(mse),
                    'mae': mean_absolute_error(y_val, y_val_pred),
                    'r2': r2_score(y_val, y_val_pred),
                    'mape': np.mean(np.abs((y_val - y_val_pred) / np.where(y_val == 0, 1e-10, y_val))) * 100,
                    'direction_accuracy': np.mean((np.sign(y_val[1:] - y_val[:-1]) == np.sign(y_val_pred[1:] - y_val_pred[:-1])).astype(int)),
                    'max_ape': np.max(np.abs((y_val - y_val_pred) / np.where(y_val == 0, 1e-10, y_val))) * 100
                }
            }
            with open('best_linear_params.pkl', 'wb') as f:
                pickle.dump(best_params, f)
            pickle.dump(linear_model, open('best_linear_model.pkl', 'wb'))

    return linear_model, history

# Prediction and evaluation functions
def make_predictions(model, data, feature_scaler, target_scaler, time_steps=60):
    features = data.drop(columns=['Date', 'Window', 'Close', 'Adj Close'])
    features_scaled = feature_scaler.transform(features)
    X = []
    for i in range(time_steps, len(features_scaled)):
        X.append(features_scaled[i-time_steps:i])
    X = np.array(X)
    predictions = model.predict(X)
    predictions_2d = predictions.reshape(-1, 1)
    predictions_rescaled = target_scaler.inverse_transform(predictions_2d)
    predictions_rescaled = predictions_rescaled.reshape(predictions.shape)
    return predictions_rescaled

def evaluate_model_performance(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-10, y_true))) * 100
    logging.info(f'Mean Squared Error (MSE): {mse}')
    logging.info(f'Root Mean Squared Error (RMSE): {rmse}')
    logging.info(f'Mean Absolute Error (MAE): {mae}')
    logging.info(f'R² Score: {r2}')
    logging.info(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'R² Score: {r2}')
    print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')
    return mse, rmse, mae, r2, mape

def evaluate_direction_accuracy(y_true, y_pred):
    direction_accuracy = np.mean((np.sign(y_true[1:] - y_true[:-1]) == np.sign(y_pred[1:] - y_pred[:-1])).astype(int))
    logging.info(f'Direction Accuracy: {direction_accuracy:.2f}')
    print(f'Direction Accuracy: {direction_accuracy:.2f}')
    return direction_accuracy

def evaluate_max_ape(y_true, y_pred):
    max_ape = np.max(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-10, y_true))) * 100
    logging.info(f'Maximum Absolute Percentage Error (MaxAPE): {max_ape:.2f}%')
    print(f'Maximum Absolute Percentage Error (MaxAPE): {max_ape:.2f}%')
    return max_ape

def evaluate_model(model, test_data, feature_scaler, target_scaler, time_steps=60):
    features = test_data.drop(columns=['Date', 'Window', 'Close', 'Adj Close'])
    features_scaled = feature_scaler.transform(features)
    X = []
    for i in range(time_steps, len(features_scaled)):
        X.append(features_scaled[i-time_steps:i])
    X = np.array(X)
    predictions = model.predict(X)
    predictions_2d = predictions.reshape(-1, 1)
    predictions_rescaled = target_scaler.inverse_transform(predictions_2d)
    predictions_rescaled = predictions_rescaled.reshape(predictions.shape)
    test_data = test_data.copy()
    test_data['Predicted_Close'] = np.nan
    test_data.iloc[time_steps:, test_data.columns.get_loc('Predicted_Close')] = predictions_rescaled.flatten()
    test_data['Error'] = test_data['Predicted_Close'] - test_data['Close']
    mse = (test_data['Error'] ** 2).mean()
    y_true = test_data['Close'][time_steps:]
    y_pred = test_data['Predicted_Close'][time_steps:]
    evaluate_model_performance(y_true, y_pred)
    evaluate_direction_accuracy(y_true, y_pred)
    evaluate_max_ape(y_true, y_pred)
    return mse, y_true, y_pred

def train_lstm_with_hyperparameter_search(X_train, y_train, X_val, y_val, input_shape, target_scaler, epochs=100, batch_size=32):
    return train_final_model(X_train, y_train, X_val, y_val, input_shape, target_scaler, epochs=epochs, batch_size=batch_size, model_type='lstm')

# Cross-validation function
def cross_validate_model(X, y, input_shape, target_scaler, model_type='gru', epochs=100, batch_size=32, splits=5):
    kf = KFold(n_splits=splits)
    all_histories = []
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        model, history = train_final_model(X_train, y_train, X_val, y_val, input_shape, target_scaler, epochs=epochs, batch_size=batch_size, model_type=model_type)
        all_histories.append(history.history)
    return all_histories
