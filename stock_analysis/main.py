import sys
import os
import logging
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.model_selection import train_test_split
import numpy as np
import talib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle

# Add project root to Python module search path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import functions from analysis/model_training.py
from stock_analysis.analysis.rolling_window import rolling_aggregate
from stock_analysis.analysis.feature_aggregation import calculate_sentiment_features
from stock_analysis.utils.data_loader import load_data, preprocess_combined_news
from stock_analysis.analysis.model_training import (
    prepare_data, train_final_model, train_lstm_with_hyperparameter_search,
    train_svm_with_hyperparameter_optimization, cross_validate_model, CustomMetrics, DynamicEarlyStopping,
    train_linear_regression_with_hyperparameter_search  # Replacing Logistic Regression function
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def analyze_sentiment(news_data):
    model_name = "./finbert_model"  # Replace with your model path
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, return_all_scores=True)

    sentiments = []
    for idx, news in enumerate(tqdm(news_data['NewsHeadline'], desc="Analyzing Sentiment")):
        sentiment_scores = sentiment_pipeline(news)
        sentiment_dict = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
        for score in sentiment_scores[0]:
            sentiment_dict[score['label'].capitalize()] = score['score']
        sentiments.append(sentiment_dict)
    sentiment_df = pd.DataFrame(sentiments)
    return pd.concat([news_data.reset_index(drop=True), sentiment_df], axis=1)

def merge_and_analyze(reddit_news, combined_news_long):
    combined_news_grouped = combined_news_long.groupby('Date')['NewsHeadline'].apply(list).reset_index()
    reddit_news_grouped = reddit_news.groupby('Date')['NewsHeadline'].apply(list).reset_index()

    # Ensure date column types are consistent
    combined_news_grouped['Date'] = pd.to_datetime(combined_news_grouped['Date'])
    reddit_news_grouped['Date'] = pd.to_datetime(reddit_news_grouped['Date'])

    merged_news = pd.merge(combined_news_grouped, reddit_news_grouped, on='Date', how='outer', suffixes=('_combined', '_reddit'))
    merged_news['AllNews'] = merged_news.apply(
        lambda row: (row['NewsHeadline_combined'] if isinstance(row['NewsHeadline_combined'], list) else []) + 
                    (row['NewsHeadline_reddit'] if isinstance(row['NewsHeadline_reddit'], list) else []), axis=1)
    merged_news = merged_news[['Date', 'AllNews']]
    merged_news = merged_news.explode('AllNews').dropna().reset_index(drop=True)
    merged_news = merged_news.rename(columns={'AllNews': 'NewsHeadline'})
    # Filter out dates with less than 20 news items
    news_count = merged_news.groupby('Date').size().reset_index(name='count')
    valid_dates = news_count[news_count['count'] >= 20]['Date']
    merged_news = merged_news[merged_news['Date'].isin(valid_dates)]
    return merged_news

def calculate_sentiment_indicators(sentiment_data):
    sentiment_data['Bullishness'] = np.log(
        (1 + sentiment_data['Positive'] + 0.5 * sentiment_data['Neutral']) /
        (1 + sentiment_data['Negative'] + 0.5 * sentiment_data['Neutral'])
    )
    sentiment_data['MessageVolume'] = np.log(
        sentiment_data['Positive'] + sentiment_data['Negative'] + sentiment_data['Neutral']
    )
    sentiment_data['Agreement'] = 1 - np.sqrt(
        1 - ((sentiment_data['Positive'] - sentiment_data['Negative']) /
             (sentiment_data['Positive'] + sentiment_data['Negative'] + sentiment_data['Neutral']))
    )
    return sentiment_data

def calculate_rolling_indicators(news_data, windows=[1, 3, 7, 15, 30]):
    results = {}
    news_data['Date'] = pd.to_datetime(news_data['Date'])
    for window in windows:
        rolling_data = news_data.set_index('Date').rolling(window=window, min_periods=1).mean()
        rolling_data = calculate_sentiment_indicators(rolling_data)
        results[window] = rolling_data.reset_index()
    return results

def calculate_feature_importance(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    importance_scores = model.feature_importances_
    return importance_scores

def evaluate_model_performance(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-10, y_true))) * 100  # Handle division by zero
    logging.info(f'Mean Squared Error (MSE): {mse}')
    logging.info(f'Root Mean Squared Error (RMSE): {rmse}')
    logging.info(f'Mean Absolute Error (MAE): {mae}')
    logging.info(f'R² Score: {r2}')
    logging.info(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')
    return mse, rmse, mae, r2, mape

def evaluate_direction_accuracy(y_true, y_pred):
    direction_accuracy = np.mean((np.sign(y_true[1:] - y_true[:-1]) == np.sign(y_pred[1:] - y_pred[:-1])).astype(int))
    logging.info(f'Direction Accuracy: {direction_accuracy:.2f}')
    return direction_accuracy

def evaluate_max_ape(y_true, y_pred):
    max_ape = np.max(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-10, y_true))) * 100  # Handle division by zero
    logging.info(f'Maximum Absolute Percentage Error (MaxAPE): {max_ape:.2f}%')
    return max_ape

def train_and_evaluate_other_models(train_data, test_data):
    y_train = train_data['Close']
    y_test = test_data['Close']
    exog_train = train_data.drop(columns=['Date', 'Close'])
    exog_test = test_data.drop(columns=['Date', 'Close'])

    # ES model
    es_model = ExponentialSmoothing(y_train, trend='add', seasonal='add', seasonal_periods=12).fit()
    es_pred = es_model.forecast(len(y_test))

    # ARIMA model
    arima_model = ARIMA(y_train, order=(5, 1, 0)).fit()
    arima_pred = arima_model.forecast(steps=len(y_test))

    # SARIMA model
    sarima_model = SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), exog=exog_train).fit(disp=False)
    sarima_pred = sarima_model.forecast(steps=len(y_test), exog=exog_test)

    return y_test, es_pred, arima_pred, sarima_pred

def add_technical_indicators(data):
    data['SMA'] = talib.SMA(data['Close'], timeperiod=30)
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
    data['MACD'], data['MACD_signal'], data['MACD_hist'] = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    data.fillna(method='backfill', inplace=True)  # Fill missing values
    return data

def smooth_data(data, window_size=5):
    data['Close'] = data['Close'].rolling(window=window_size).mean()
    data = data.dropna()
    return data

def load_best_params(model_type):
    params_path = f'best_{model_type}_params.pkl'
    if os.path.exists(params_path):
        with open(params_path, 'rb') as f:
            best_params = pickle.load(f)
        return best_params
    return None

def train_models(model_types_to_train):
    logging.info("Loading final dataset for model training and evaluation...")
    final_data = pd.read_csv('./stock_analysis/data/processed_data.csv')

    logging.info("Splitting dataset into training and test sets...")
    train_data, test_data = train_test_split(final_data, test_size=0.2, shuffle=False)

    logging.info("Preparing data for model training and evaluation...")
    X_train, y_train, feature_scaler, target_scaler = prepare_data(train_data, time_steps=60)
    X_val, y_val, _, _ = prepare_data(test_data, time_steps=60)
    input_shape = (60, X_train.shape[2])

    histories = {}
    
    if 'gru' in model_types_to_train:
        logging.info("Cross-validating and training the final GRU model...")
        gru_histories = cross_validate_model(X_train, y_train, input_shape, target_scaler, model_type='gru', epochs=100, batch_size=32, splits=5)

        for i, history in enumerate(gru_histories):
            pd.DataFrame(history).to_csv(f'./stock_analysis/data/gru_history_fold_{i}.csv', index=False)

        histories['gru'] = gru_histories

    if '12gru' in model_types_to_train:
        logging.info("Building and training the final 12-layer GRU model...")
        final_12gru_model, final_12gru_history = train_final_model(X_train, y_train, X_val, y_val, input_shape, target_scaler, epochs=100, batch_size=32, model_type='12gru', additional_callbacks=[])
        pd.DataFrame(final_12gru_history.history).to_csv('./stock_analysis/data/final_12gru_history.csv', index=False)

        logging.info("Evaluating the final 12-layer GRU model...")
        y_val_pred = final_12gru_model.predict(X_val)
        y_val_true = target_scaler.inverse_transform(y_val)
        y_val_pred = y_val_pred.reshape((y_val_pred.shape[0], y_val_pred.shape[1]))
        y_val_pred = target_scaler.inverse_transform(y_val_pred)
        y_val_true, y_val_pred = y_val_true.flatten(), y_val_pred.flatten()
        min_length = min(len(y_val_true), len(y_val_pred))
        y_val_true, y_val_pred = y_val_true[:min_length], y_val_pred[:min_length]
        pd.DataFrame({'Close': y_val_true, 'Predicted_Close': y_val_pred}).to_csv('./stock_analysis/data/y_true_12gru.csv', index=False)
        final_12gru_mse, final_12gru_rmse, final_12gru_mae, final_12gru_r2, final_12gru_mape = evaluate_model_performance(y_val_true, y_val_pred)
        final_12gru_direction_accuracy = evaluate_direction_accuracy(y_val_true, y_val_pred)
        final_12gru_max_ape = evaluate_max_ape(y_val_true, y_val_pred)

        histories['12gru'] = final_12gru_history

    if 'lstm' in model_types_to_train:
        logging.info("Cross-validating and training LSTM model...")
        lstm_histories = cross_validate_model(X_train, y_train, input_shape, target_scaler, model_type='lstm', epochs=100, batch_size=32, splits=5)

        for i, history in enumerate(lstm_histories):
            pd.DataFrame(history).to_csv(f'./stock_analysis/data/lstm_history_fold_{i}.csv', index=False)

        histories['lstm'] = lstm_histories

    if 'svm' in model_types_to_train:
        logging.info("Training and evaluating SVM model...")
        svm_model, svm_history = train_svm_with_hyperparameter_optimization(X_train, y_train, X_val, y_val, target_scaler, epochs=100)
        y_val_pred = svm_model.predict(X_val.reshape(X_val.shape[0], -1))
        y_val_true = target_scaler.inverse_transform(y_val)
        y_val_pred = target_scaler.inverse_transform(y_val_pred.reshape(-1, 1))
        y_val_true, y_val_pred = y_val_true.flatten(), y_val_pred.flatten()
        min_length = min(len(y_val_true), len(y_val_pred))
        y_val_true, y_val_pred = y_val_true[:min_length], y_val_pred[:min_length]
        pd.DataFrame({'Close': y_val_true, 'Predicted_Close': y_val_pred}).to_csv('./stock_analysis/data/y_true_svm.csv', index=False)
        svm_mse, svm_rmse, svm_mae, svm_r2, svm_mape = evaluate_model_performance(y_val_true, y_val_pred)
        svm_direction_accuracy = evaluate_direction_accuracy(y_val_true, y_val_pred)
        svm_max_ape = evaluate_max_ape(y_val_true, y_val_pred)

        histories['svm'] = svm_history

    if 'linear' in model_types_to_train:
        logging.info("Training and evaluating Linear Regression model...")
        linear_model, linear_history = train_linear_regression_with_hyperparameter_search(X_train, y_train, X_val, y_val, target_scaler, epochs=100)
        y_val_pred_linear = linear_model.predict(X_val.reshape(X_val.shape[0], -1))
        y_val_true = target_scaler.inverse_transform(y_val).flatten()  # Ensure y_val_true is a 1D array
        y_val_pred_linear = target_scaler.inverse_transform(y_val_pred_linear.reshape(-1, 1)).flatten()  # Ensure y_val_pred_linear is a 1D array
        pd.DataFrame({'Close': y_val_true, 'Predicted_Close': y_val_pred_linear}).to_csv('./stock_analysis/data/y_true_linear.csv', index=False)
        linear_mse, linear_rmse, linear_mae, linear_r2, linear_mape = evaluate_model_performance(y_val_true, y_val_pred_linear)
        linear_direction_accuracy = evaluate_direction_accuracy(y_val_true, y_val_pred_linear)
        linear_max_ape = evaluate_max_ape(y_val_true, y_val_pred_linear)

        histories['linear'] = linear_history

    logging.info("Training and evaluating ES, ARIMA, and SARIMA models...")
    y_test, es_pred, arima_pred, sarima_pred = train_and_evaluate_other_models(train_data, test_data)
    es_mse, es_rmse, es_mae, es_r2, es_mape = evaluate_model_performance(y_test, es_pred)
    es_direction_accuracy = evaluate_direction_accuracy(y_test, es_pred)
    es_max_ape = evaluate_max_ape(y_test, es_pred)
    arima_mse, arima_rmse, arima_mae, arima_r2, arima_mape = evaluate_model_performance(y_test, arima_pred)
    arima_direction_accuracy = evaluate_direction_accuracy(y_test, arima_pred)
    arima_max_ape = evaluate_max_ape(y_test, arima_pred)
    sarima_mse, sarima_rmse, sarima_mae, sarima_r2, sarima_mape = evaluate_model_performance(y_test, sarima_pred)
    sarima_direction_accuracy = evaluate_direction_accuracy(y_test, sarima_pred)
    sarima_max_ape = evaluate_max_ape(y_test, sarima_pred)

    other_params = {
        'linear': {
            'history': histories.get('linear', None),
            'mse': linear_mse if 'linear' in model_types_to_train else None,
            'rmse': linear_rmse if 'linear' in model_types_to_train else None,
            'mae': linear_mae if 'linear' in model_types_to_train else None,
            'r2': linear_r2 if 'linear' in model_types_to_train else None,
            'mape': linear_mape if 'linear' in model_types_to_train else None,
            'direction_accuracy': linear_direction_accuracy if 'linear' in model_types_to_train else None,
            'max_ape': linear_max_ape if 'linear' in model_types_to_train else None
        },
        'es': {
            'mse': es_mse,
            'rmse': es_rmse,
            'mae': es_mae,
            'r2': es_r2,
            'mape': es_mape,
            'direction_accuracy': es_direction_accuracy,
            'max_ape': es_max_ape
        },
        'arima': {
            'mse': arima_mse,
            'rmse': arima_rmse,
            'mae': arima_mae,
            'r2': arima_r2,
            'mape': arima_mape,
            'direction_accuracy': arima_direction_accuracy,
            'max_ape': arima_max_ape
        },
        'sarima': {
            'mse': sarima_mse,
            'rmse': sarima_rmse,
            'mae': sarima_mae,
            'r2': sarima_r2,
            'mape': sarima_mape,
            'direction_accuracy': sarima_direction_accuracy,
            'max_ape': sarima_max_ape
        }
    }

    with open('other_model_params.pkl', 'wb') as f:
        pickle.dump(other_params, f)

    # Print ES, ARIMA, and SARIMA evaluation metrics
    logging.info("Exponential Smoothing (ES) Model:")
    logging.info(f"MSE: {es_mse}, RMSE: {es_rmse}, MAE: {es_mae}, R²: {es_r2}, MAPE: {es_mape}, Direction Accuracy: {es_direction_accuracy}, MaxAPE: {es_max_ape}")
    
    logging.info("ARIMA Model:")
    logging.info(f"MSE: {arima_mse}, RMSE: {arima_rmse}, MAE: {arima_mae}, R²: {arima_r2}, MAPE: {arima_mape}, Direction Accuracy: {arima_direction_accuracy}, MaxAPE: {arima_max_ape}")
    
    logging.info("SARIMA Model:")
    logging.info(f"MSE: {sarima_mse}, RMSE: {sarima_rmse}, MAE: {sarima_mae}, R²: {sarima_r2}, MAPE: {sarima_mape}, Direction Accuracy: {sarima_direction_accuracy}, MaxAPE: {sarima_max_ape}")

    logging.info("All tasks completed successfully.")

def clean_news_data(news_data):
    logging.info("Cleaning news data...")
    news_data.drop_duplicates(subset=['Date', 'NewsHeadline'], inplace=True)
    news_data.dropna(subset=['Date', 'NewsHeadline'], inplace=True)
    return news_data

def main():
    news_data_with_sentiment_file = 'stock_analysis/data/news_data_with_sentiment.csv'
    cleaned_news_data_with_sentiment_file = 'stock_analysis/data/cleaned_news_data_with_sentiment.csv'
    daily_sentiment_file = 'stock_analysis/data/daily_sentiment_with_indicators.csv'
    rolling_files = {
        1: 'stock_analysis/data/rolling_sentiment_1.csv',
        3: 'stock_analysis/data/rolling_sentiment_3.csv',
        7: 'stock_analysis/data/rolling_sentiment_7.csv',
        15: 'stock_analysis/data/rolling_sentiment_15.csv',
        30: 'stock_analysis/data/rolling_sentiment_30.csv'
    }
    processed_data_file = 'stock_analysis/data/processed_data.csv'

    if (os.path.exists(processed_data_file)):
        logging.info("Loading existing processed_data.csv...")
        final_data = pd.read_csv(processed_data_file)
    else:
        if not os.path.exists(daily_sentiment_file):
            if not os.path.exists(cleaned_news_data_with_sentiment_file):
                if not os.path.exists(news_data_with_sentiment_file):
                    logging.info("Loading data...")
                    reddit_news, djia_data, combined_news = load_data()
                    logging.info("Preprocessing Combined_News_DJIA data...")
                    combined_news_long = preprocess_combined_news(combined_news)
                    logging.info("Merging news data...")
                    news_data = merge_and_analyze(reddit_news, combined_news_long)
                    news_data.to_csv('stock_analysis/data/news_data.csv', index=False)
                
                    # Analyze sentiment for original news data
                    logging.info("Analyzing sentiment for original news data...")
                    news_data_chunks = []
                    chunk_size = 1000
                    for chunk in tqdm(pd.read_csv('stock_analysis/data/news_data.csv', chunksize=chunk_size), desc="Sentiment Analysis"):
                        chunk = analyze_sentiment(chunk)
                        news_data_chunks.append(chunk)
                    news_data = pd.concat(news_data_chunks, ignore_index=True)
                    news_data.to_csv(news_data_with_sentiment_file, index=False)
                else:
                    logging.info("Loading existing news_data_with_sentiment.csv...")
                    news_data = pd.read_csv(news_data_with_sentiment_file)

                # Clean news data
                news_data = clean_news_data(news_data)
                news_data.to_csv(cleaned_news_data_with_sentiment_file, index=False)
            else:
                logging.info("Loading existing cleaned_news_data_with_sentiment.csv...")
                news_data = pd.read_csv(cleaned_news_data_with_sentiment_file)

            logging.info("Calculating daily sentiment features...")
            daily_sentiment = calculate_sentiment_features(news_data)
            daily_sentiment.to_csv('stock_analysis/data/daily_sentiment.csv', index=False)
            daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'])
            logging.info("Calculating sentiment indicators for daily sentiment...")
            daily_sentiment = calculate_sentiment_indicators(daily_sentiment)
            daily_sentiment.to_csv(daily_sentiment_file, index=False)
        else:
            logging.info("Loading existing daily sentiment with indicators file...")
            daily_sentiment = pd.read_csv(daily_sentiment_file)
        daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'])

        logging.info("Calculating rolling sentiment indicators...")
        rolling_indicators = calculate_rolling_indicators(daily_sentiment)

        logging.info("Saving rolling sentiment indicators to files...")
        for window, data in rolling_indicators.items():
            rolling_files[window] = f'stock_analysis/data/rolling_sentiment_{window}.csv'
            data.to_csv(rolling_files[window], index=False)
        
        logging.info("Merging rolling sentiment data with DJIA data...")
        reddit_news, djia_data, combined_news = load_data()
        djia_data['Date'] = pd.to_datetime(djia_data['Date'])
        merged_data_list = []
        for window, file in rolling_files.items():
            rolling_sentiment = pd.read_csv(file)
            rolling_sentiment['Date'] = pd.to_datetime(rolling_sentiment['Date'])
            merged_data = pd.merge(djia_data, rolling_sentiment, on='Date', how='inner')
            if merged_data.empty:
                logging.warning(f"merged_data for window {window} is empty!")
            merged_data['Window'] = window
            merged_data_list.append(merged_data)
        all_sentiments = pd.concat(merged_data_list, ignore_index=True)
        
        logging.info("Selecting the best window data for each date based on Bullishness with log penalty for larger windows...")
        all_sentiments['Bullishness_penalty'] = all_sentiments['Bullishness'] / np.log1p(all_sentiments['Window'])
        final_data = all_sentiments.loc[all_sentiments.groupby('Date')['Bullishness_penalty'].apply(lambda x: x.abs().idxmax())]

        logging.info("Adding technical indicators to the final dataset...")
        final_data = add_technical_indicators(final_data)
        logging.info("Smoothing data...")
        final_data = smooth_data(final_data, window_size=5)
        logging.info("Saving final dataset...")
        final_data.to_csv(processed_data_file, index=False)
    
    # Check model types to train
    model_types_to_train = []
    for model_type in ['gru', '12gru', 'lstm', 'svm', 'linear']:
        if not os.path.exists(f'best_{model_type}_params.pkl'):
            model_types_to_train.append(model_type)

    # Train models if needed
    if model_types_to_train:
        train_models(model_types_to_train)

    # Load and analyze best model parameters
    best_params = {}
    for model_type in ['12gru', 'gru', 'lstm', 'svm', 'linear']:
        best_params[model_type] = load_best_params(model_type)
    other_model_params_path = 'other_model_params.pkl'
    if os.path.exists(other_model_params_path):
        with open(other_model_params_path, 'rb') as f:
            best_params.update(pickle.load(f))
    else:
        logging.error("No other model parameters found.")

    # Print ES, ARIMA, and SARIMA evaluation metrics
    es_params = best_params.get('es', {})
    arima_params = best_params.get('arima', {})
    sarima_params = best_params.get('sarima', {})

    logging.info("Exponential Smoothing (ES) Model:")
    logging.info(f"MSE: {es_params.get('mse')}, RMSE: {es_params.get('rmse')}, MAE: {es_params.get('mae')}, R²: {es_params.get('r2')}, MAPE: {es_params.get('mape')}, Direction Accuracy: {es_params.get('direction_accuracy')}, MaxAPE: {es_params.get('max_ape')}")
    
    logging.info("ARIMA Model:")
    logging.info(f"MSE: {arima_params.get('mse')}, RMSE: {arima_params.get('rmse')}, MAE: {arima_params.get('mae')}, R²: {arima_params.get('r2')}, MAPE: {arima_params.get('mape')}, Direction Accuracy: {arima_params.get('direction_accuracy')}, MaxAPE: {arima_params.get('max_ape')}")
    
    logging.info("SARIMA Model:")
    logging.info(f"MSE: {sarima_params.get('mse')}, RMSE: {sarima_params.get('rmse')}, MAE: {sarima_params.get('mae')}, R²: {sarima_params.get('r2')}, MAPE: {sarima_params.get('mape')}, Direction Accuracy: {sarima_params.get('direction_accuracy')}, MaxAPE: {sarima_params.get('max_ape')}")

    logging.info("All tasks completed successfully.")

if __name__ == "__main__":
    main()
