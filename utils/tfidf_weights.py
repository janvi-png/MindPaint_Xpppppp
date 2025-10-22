from sklearn.feature_extraction.text import TfidfVectorizer

def compute_word_weights(sentences):
    """
    Computes TF-IDF weight per word across all sentences.
    Returns dict[word] = weight
    """
    vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b')
    vectorizer.fit(sentences)
    weights = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))
    return weights
