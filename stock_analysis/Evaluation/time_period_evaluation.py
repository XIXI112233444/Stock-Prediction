import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout, InputLayer, LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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

# Train and evaluate models for a given period
def train_and_evaluate_period(data, period, model_type, input_shape, build_model_fn, epochs=100, batch_size=32):
    train_data = data[(data['Date'] >= period[0]) & (data['Date'] <= period[1])]
    X, y, _, target_scaler = preprocess_data(train_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_model_fn(input_shape)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
    y_pred = model.predict(X_test)
    
    y_true = target_scaler.inverse_transform(y_test).flatten()
    y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    
    mse, rmse, mae, r2, mape, direction_accuracy, max_ape = evaluate_model_performance(y_true, y_pred)
    
    results_dict = {
        'Period': period,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape,
        'Direction Accuracy': direction_accuracy,
        'MaxAPE': max_ape,
        'True': y_true,
        'Predicted': y_pred
    }
    
    return results_dict

# Plot boxplots for key metrics
def plot_boxplots(results, metrics, model_types):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance Comparison Across Different Metrics')
    
    for i, metric in enumerate(metrics):
        ax = axs[i // 2, i % 2]
        metric_values = []
        for model_type in model_types:
            values = [res[metric] for res in results[model_type]]
            metric_values.append(values)
        ax.boxplot(metric_values, widths=0.6)
        ax.set_title(metric)
        ax.set_xticklabels(model_types)
        ax.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'./stock_analysis/Evaluation/time_period_evaluation_data/metrics_comparison.png')
    plt.show()

# Main evaluation function
def main():
    data_file = './stock_analysis/data/processed_data.csv'
    data = load_data(data_file)

    input_shape = (60, len(data.columns) - 4)  # Excluding 'Date', 'Close', 'Adj Close', 'Window'
    
    periods = [
        ('2009-01-01', '2011-12-31'),
        ('2012-01-01', '2013-12-31'),
        ('2014-01-01', '2015-12-31')
    ]
    
    model_builders = {
        '3-layer GRU': build_gru_model,
        'LSTM': build_lstm_model
    }

    model_types = list(model_builders.keys())
    results = {model_type: [] for model_type in model_types}
    
    all_data_exists = True
    for period in periods:
        for model_type in model_types:
            file_path = f'./stock_analysis/Evaluation/time_period_evaluation_data/{model_type}_evaluation_{period[0]}_{period[1]}.csv'
            if not os.path.exists(file_path):
                all_data_exists = False
                break
        if not all_data_exists:
            break

    if all_data_exists:
        logging.info("Loading existing results for all models and periods...")
        for period in periods:
            for model_type in model_types:
                file_path = f'./stock_analysis/Evaluation/time_period_evaluation_data/{model_type}_evaluation_{period[0]}_{period[1]}.csv'
                result_df = pd.read_csv(file_path)
                results[model_type].append(result_df.to_dict('records')[0])
    else:
        logging.info("Training and evaluating models for all periods...")
        for period in periods:
            for model_type in model_types:
                logging.info(f"Training and evaluating {model_type} model for period {period}...")
                result = train_and_evaluate_period(data, period, model_type, input_shape, model_builders[model_type])
                results[model_type].append(result)
                result_df = pd.DataFrame([result])
                result_df.to_csv(f'./stock_analysis/Evaluation/time_period_evaluation_data/{model_type}_evaluation_{period[0]}_{period[1]}.csv', index=False)
    
    key_metrics = ['MSE', 'MAE', 'R²', 'MAPE']
    plot_boxplots(results, key_metrics, model_types)

if __name__ == "__main__":
    os.makedirs('./stock_analysis/Evaluation/time_period_evaluation_data', exist_ok=True)
    main()
