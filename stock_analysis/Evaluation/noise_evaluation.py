import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import GRU, LSTM, Dense, Dropout, InputLayer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Load data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    return data

# Preprocess data
def preprocess_data(data, time_steps=60):
    feature_cols = data.columns.drop(['Date', 'Close', 'Adj Close', 'Window'])
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(data[feature_cols])
    target_scaler = MinMaxScaler()
    scaled_target = target_scaler.fit_transform(data[['Close']])

    X, y = [], []
    for i in range(time_steps, len(scaled_features)):
        X.append(scaled_features[i-time_steps:i])
        y.append(scaled_target[i])
    X, y = np.array(X), np.array(y)
    return X, y, scaler, target_scaler

# Add noise to data
def add_noise(data, noise_level=0.01):
    noisy_data = data.copy()
    noise = np.random.normal(0, noise_level, noisy_data['Close'].shape)
    noisy_data['Close'] += noise
    return noisy_data

# Evaluate model performance
def evaluate_model_performance(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-10, y_true))) * 100
    direction_accuracy = np.mean((np.sign(y_true[1:] - y_true[:-1]) == np.sign(y_pred[1:] - y_pred[:-1])).astype(int))
    max_ape = np.max(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-10, y_true))) * 100
    return mse, rmse, mae, r2, mape, direction_accuracy, max_ape

# Train and evaluate models with noise
def train_and_evaluate_with_noise(data, model_type, input_shape, noise_levels, build_model_fn, epochs=100, batch_size=32):
    results = []
    for noise_level in noise_levels:
        noisy_data = add_noise(data, noise_level=noise_level)
        X, y, _, target_scaler = preprocess_data(noisy_data)
        
        model = build_model_fn(input_shape)
        model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
        y_pred = model.predict(X)
        
        y_true = target_scaler.inverse_transform(y).flatten()
        y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        mse, rmse, mae, r2, mape, direction_accuracy, max_ape = evaluate_model_performance(y_true, y_pred)
        result_dict = {
            'Model': model_type,
            'Noise Level': noise_level,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'MAPE': mape,
            'Direction Accuracy': direction_accuracy,
            'MaxAPE': max_ape
        }
        results.append(result_dict)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'./stock_analysis/Evaluation/noise_evaluation_data/{model_type}_evaluation_noise.csv', index=False)
    return results_df

# Define models
def build_gru_model(input_shape):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    for units in [320, 320, 320]:
        model.add(GRU(units=units, dropout=0.0, return_sequences=True))
    model.add(GRU(units=320, dropout=0.0, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    for units in [320, 320, 320]:
        model.add(LSTM(units=units, dropout=0.0, return_sequences=True))
    model.add(LSTM(units=320, dropout=0.0, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Plot boxplots for key metrics
def plot_boxplots(results, metrics, model_types):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance Comparison Across Different Noise Levels')
    
    for i, metric in enumerate(metrics):
        ax = axs[i // 2, i % 2]
        metric_values = []
        for model_type in model_types:
            values = [res[metric] for res in results if res['Model'] == model_type]
            metric_values.append(values)
        ax.boxplot(metric_values, widths=0.6)
        ax.set_title(metric)
        ax.set_xticklabels(model_types)
        ax.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'./stock_analysis/Evaluation/noise_evaluation_data/metrics_noise_comparison.png')
    plt.show()

# Main evaluation function
def main():
    data_file = './stock_analysis/data/processed_data.csv'
    data = load_data(data_file)

    input_shape = (60, len(data.columns) - 4)  # Excluding 'Date', 'Close', 'Adj Close', 'Window'
    
    noise_levels = [0.01, 0.05, 0.1]
    
    model_builders = {
        '3-layer GRU': build_gru_model,
        'LSTM': build_lstm_model
    }

    model_types = list(model_builders.keys())
    results = []

    all_data_exists = True
    for model_type in model_types:
        file_path = f'./stock_analysis/Evaluation/noise_evaluation_data/{model_type}_evaluation_noise.csv'
        if not os.path.exists(file_path):
            all_data_exists = False
            break

    if all_data_exists:
        logging.info("Loading existing results for all models and noise levels...")
        for model_type in model_types:
            file_path = f'./stock_analysis/Evaluation/noise_evaluation_data/{model_type}_evaluation_noise.csv'
            results_df = pd.read_csv(file_path)
            results.extend(results_df.to_dict('records'))
    else:
        logging.info("Training and evaluating models for all noise levels...")
        for model_type in model_types:
            logging.info(f"Training and evaluating {model_type} model with noise levels {noise_levels}...")
            results_df = train_and_evaluate_with_noise(data, model_type, input_shape, noise_levels, model_builders[model_type])
            results.extend(results_df.to_dict('records'))

    key_metrics = ['MSE', 'MAE', 'R²', 'MAPE']
    plot_boxplots(results, key_metrics, model_types)

if __name__ == "__main__":
    os.makedirs('./stock_analysis/Evaluation/noise_evaluation_data', exist_ok=True)
    main()
