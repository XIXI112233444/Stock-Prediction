import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import os

# Ensure that the directory where the chart is saved exists
output_dir = './stock_analysis/data/Figure'
os.makedirs(output_dir, exist_ok=True)

data_12gru = pd.read_csv('./stock_analysis/data/y_true_12gru.csv')
data_gru = pd.read_csv('./stock_analysis/data/y_true_gru.csv')
data_lstm = pd.read_csv('./stock_analysis/data/y_true_lstm.csv')
data_svm = pd.read_csv('./stock_analysis/data/y_true_svm.csv')

y_true_12gru = data_12gru['Close'].values
y_pred_12gru = data_12gru['Predicted_Close'].values

y_true_gru = data_gru['Close'].values
y_pred_gru = data_gru['Predicted_Close'].values

y_true_lstm = data_lstm['Close'].values
y_pred_lstm = data_lstm['Predicted_Close'].values

y_true_svm = data_svm['Close'].values
y_pred_svm = data_svm['Predicted_Close'].values

# calculation error
error_12gru = y_true_12gru - y_pred_12gru
error_gru = y_true_gru - y_pred_gru
error_lstm = y_true_lstm - y_pred_lstm
error_svm = y_true_svm - y_pred_svm

# Plotting the residuals
plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1)
plt.scatter(y_pred_gru, error_gru, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Close')
plt.ylabel('Residuals')
plt.title('GRU Residuals')

plt.subplot(2, 2, 2)
plt.scatter(y_pred_lstm, error_lstm, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Close')
plt.ylabel('Residuals')
plt.title('LSTM Residuals')

plt.subplot(2, 2, 3)
plt.scatter(y_pred_12gru, error_12gru, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Close')
plt.ylabel('Residuals')
plt.title('12-layer GRU Residuals')

plt.subplot(2, 2, 4)
plt.scatter(y_pred_svm, error_svm, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Close')
plt.ylabel('Residuals')
plt.title('SVM Residuals')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'residuals_comparison.png'))
plt.show()

# Plotting a scatter plot of actual versus predicted values
plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1)
plt.scatter(y_true_gru, y_pred_gru, alpha=0.6)
plt.plot([min(y_true_gru), max(y_true_gru)], [min(y_true_gru), max(y_true_gru)], color='red')
plt.xlabel('Actual Close')
plt.ylabel('Predicted Close')
plt.title('GRU Actual vs Predicted')

plt.subplot(2, 2, 2)
plt.scatter(y_true_lstm, y_pred_lstm, alpha=0.6)
plt.plot([min(y_true_lstm), max(y_true_lstm)], [min(y_true_lstm), max(y_true_lstm)], color='red')
plt.xlabel('Actual Close')
plt.ylabel('Predicted Close')
plt.title('LSTM Actual vs Predicted')

plt.subplot(2, 2, 3)
plt.scatter(y_true_12gru, y_pred_12gru, alpha=0.6)
plt.plot([min(y_true_12gru), max(y_true_12gru)], [min(y_true_12gru), max(y_true_12gru)], color='red')
plt.xlabel('Actual Close')
plt.ylabel('Predicted Close')
plt.title('12-layer GRU Actual vs Predicted')

plt.subplot(2, 2, 4)
plt.scatter(y_true_svm, y_pred_svm, alpha=0.6)
plt.plot([min(y_true_svm), max(y_true_svm)], [min(y_true_svm), max(y_true_svm)], color='red')
plt.xlabel('Actual Close')
plt.ylabel('Predicted Close')
plt.title('SVM Actual vs Predicted')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'scatter_comparison.png'))
plt.show()

# Plotting the error distribution
plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1)
plt.hist(error_gru, bins=30, alpha=0.7, label='GRU')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('GRU Error Distribution')
plt.legend()

plt.subplot(2, 2, 2)
plt.hist(error_lstm, bins=30, alpha=0.7, label='LSTM')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('LSTM Error Distribution')
plt.legend()

plt.subplot(2, 2, 3)
plt.hist(error_12gru, bins=30, alpha=0.7, label='12-layer GRU')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('12-layer GRU Error Distribution')
plt.legend()

plt.subplot(2, 2, 4)
plt.hist(error_svm, bins=30, alpha=0.7, label='SVM')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('SVM Error Distribution')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'error_distribution_comparison.png'))
plt.show()

# Plotting time series
plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1)
plt.plot(y_true_gru, label='Actual')
plt.plot(y_pred_gru, label='GRU Prediction')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.title('GRU Time Series')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(y_true_lstm, label='Actual')
plt.plot(y_pred_lstm, label='LSTM Prediction')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.title('LSTM Time Series')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(y_true_12gru, label='Actual')
plt.plot(y_pred_12gru, label='12-layer GRU Prediction')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.title('12-layer GRU Time Series')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(y_true_svm, label='Actual')
plt.plot(y_pred_svm, label='SVM Prediction')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.title('SVM Time Series')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'time_series_comparison.png'))
plt.show()

# Calculate and show the mean square error (MSE) for each model
mse_12gru = mean_squared_error(y_true_12gru, y_pred_12gru)
mse_gru = mean_squared_error(y_true_gru, y_pred_gru)
mse_lstm = mean_squared_error(y_true_lstm, y_pred_lstm)
mse_svm = mean_squared_error(y_true_svm, y_pred_svm)

print(f'12-layer GRU Model MSE: {mse_12gru}')
print(f'GRU Model MSE: {mse_gru}')
print(f'LSTM Model MSE: {mse_lstm}')
print(f'SVM Model MSE: {mse_svm}')