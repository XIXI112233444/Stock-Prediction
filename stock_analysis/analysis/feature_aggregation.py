import numpy as np
import pandas as pd

def calculate_sentiment_features(news_data):
    # Calculate the average of sentiment scores
    sentiment_features = news_data.groupby('Date').agg({
        'Positive': 'mean',
        'Negative': 'mean',
        'Neutral': 'mean'
    }).reset_index()

    return sentiment_features
