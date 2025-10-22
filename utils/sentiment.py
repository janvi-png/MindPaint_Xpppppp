from textblob import TextBlob

def get_sentiment_scores(words):
    """
    Returns sentiment polarity (-1 to 1) for each word.
    """
    return [TextBlob(w).sentiment.polarity for w in words]
