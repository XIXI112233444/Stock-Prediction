import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

def analyze_sentiment(news_data):
    tokenizer = AutoTokenizer.from_pretrained("finbert_model")
    model = AutoModelForSequenceClassification.from_pretrained("finbert_model")
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    def sentiment_scores(text):
        result = nlp(text)
        scores = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
        for res in result:
            if res['label'] == 'positive':
                scores['Positive'] = res['score']
            elif res['label'] == 'negative':
                scores['Negative'] = res['score']
            elif res['label'] == 'neutral':
                scores['Neutral'] = res['score']
        return scores

    scores = news_data['NewsHeadline'].apply(lambda x: pd.Series(sentiment_scores(x)))
    news_data = pd.concat([news_data, scores], axis=1)
    return news_data
