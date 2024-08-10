import pandas as pd
import numpy as np

def rolling_aggregate(df, window=3):
    # Ensure that the 'Date' column exists and is of datetime type
    if 'Date' not in df.columns:
        raise ValueError("The input dataframe must contain a 'Date' column.")
    df['Date'] = pd.to_datetime(df['Date'])

    df = df.sort_values('Date')

    # 计算滚动窗口聚合
    rolling_data = df.rolling(window=window, min_periods=1).agg({
        'Positive': 'mean',
        'Negative': 'mean',
        'Neutral': 'mean'
    }).reset_index(drop=True)

    # Retention of date information
    rolling_data['Date'] = df['Date']
    
    return rolling_data
