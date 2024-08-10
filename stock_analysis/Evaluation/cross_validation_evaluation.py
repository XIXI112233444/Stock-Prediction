import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Define constants
METRICS = ['val_mse', 'val_rmse', 'val_mae', 'val_r2', 'val_mape', 'val_max_ape', 'val_direction_accuracy']
FOLD_FILES = {
    'GRU': [
        './stock_analysis/data/gru_history_fold_0.csv',
        './stock_analysis/data/gru_history_fold_1.csv',
        './stock_analysis/data/gru_history_fold_2.csv',
        './stock_analysis/data/gru_history_fold_3.csv',
        './stock_analysis/data/gru_history_fold_4.csv'
    ],
    'LSTM': [
        './stock_analysis/data/lstm_history_fold_0.csv',
        './stock_analysis/data/lstm_history_fold_1.csv',
        './stock_analysis/data/lstm_history_fold_2.csv',
        './stock_analysis/data/lstm_history_fold_3.csv',
        './stock_analysis/data/lstm_history_fold_4.csv'
    ]
}

# Compute mean and std for each fold
def compute_fold_statistics(file_paths):
    fold_means = []
    fold_stds = []

    for file_path in file_paths:
        data = pd.read_csv(file_path)
        fold_means.append(data[METRICS].mean())
        fold_stds.append(data[METRICS].std())

    fold_means_df = pd.DataFrame(fold_means)
    fold_stds_df = pd.DataFrame(fold_stds)

    return fold_means_df, fold_stds_df

# Aggregate mean and std across folds
def aggregate_statistics(fold_means_df, fold_stds_df):
    aggregated_means = fold_means_df.mean()
    aggregated_stds = fold_stds_df.mean()
    
    result_df = pd.DataFrame({
        'mean': aggregated_means,
        'std': aggregated_stds
    })

    return result_df

# Plot boxplots for key metrics
def plot_boxplots(all_fold_means, model_types, metrics):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Cross-Validation Model Performance Comparison')

    for i, metric in enumerate(metrics):
        ax = axs[i // 2, i % 2]
        metric_values = [all_fold_means[model][metric] for model in model_types]
        ax.boxplot(metric_values, widths=0.6)
        ax.set_title(metric)
        ax.set_xticklabels(model_types)
        ax.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('./stock_analysis/Evaluation/cross_validation_evaluation_data/cross_validation_metrics_comparison.png')
    plt.show()

# Main evaluation function
def main():
    os.makedirs('./stock_analysis/Evaluation/cross_validation_evaluation_data', exist_ok=True)
    
    all_fold_means = {model: [] for model in FOLD_FILES.keys()}
    all_fold_stds = {model: [] for model in FOLD_FILES.keys()}

    for model_type, file_paths in FOLD_FILES.items():
        logging.info(f"Processing {model_type} model...")
        fold_means_df, fold_stds_df = compute_fold_statistics(file_paths)
        aggregated_df = aggregate_statistics(fold_means_df, fold_stds_df)
        
        aggregated_df.to_csv(f'./stock_analysis/Evaluation/cross_validation_evaluation_data/{model_type}_cross_validation_evaluation.csv', index=True)
        
        all_fold_means[model_type] = fold_means_df
        all_fold_stds[model_type] = fold_stds_df
    
    key_metrics = ['val_mse', 'val_mae', 'val_r2', 'val_mape']
    plot_boxplots(all_fold_means, list(FOLD_FILES.keys()), key_metrics)

if __name__ == "__main__":
    main()
