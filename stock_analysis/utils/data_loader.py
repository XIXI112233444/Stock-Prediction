import pandas as pd

def load_data():
    reddit_news = pd.read_csv('stock_analysis/data/RedditNews.csv')
    reddit_news = reddit_news.rename(columns={'News': 'NewsHeadline'})  # 确保列名一致
    djia_data = pd.read_csv('stock_analysis/data/DJIA_table.csv')
    combined_news = pd.read_csv('stock_analysis/data/Combined_News_DJIA.csv')
    return reddit_news, djia_data, combined_news

def preprocess_combined_news(combined_news):
    combined_news_long = pd.melt(combined_news, id_vars=['Date'], value_vars=[f'Top{i}' for i in range(1, 26)], var_name='Top', value_name='NewsHeadline')
    combined_news_long = combined_news_long.drop(columns=['Top'])
    combined_news_long['Date'] = pd.to_datetime(combined_news_long['Date'], errors='coerce')
    combined_news_long = combined_news_long.dropna(subset=['Date'])
    return combined_news_long
