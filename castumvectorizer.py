import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Optional, Dict


class SentimentWeightedTfidf(BaseEstimator, TransformerMixin):
    """
    Custom TF-IDF transformer for Amazon reviews with preprocessing integration.
    """

    def __init__(
            self,
            max_features: int = 5000,
            min_df: int = 2,
            sentiment_weight: float = 2.0,
            sentiment_words: Optional[List[str]] = None
    ):
        self.sentiment_words = sentiment_words or [
            # Product-specific
            'great', 'excellent', 'perfect', 'best', 'love', 'amazing',
            'good', 'nice', 'worth', 'happy', 'disappointed', 'terrible',
            'horrible', 'waste', 'poor', 'bad', 'worst', 'broken',

            # Quality indicators
            'quality', 'well', 'better', 'works', 'working', 'worked',
            'break', 'broken', 'sturdy', 'durable', 'cheap', 'expensive',

            # Amazon-specific
            'shipping', 'delivery', 'arrived', 'recommend', 'purchase',
            'return', 'sent', 'refund', 'customer', 'service', 'price',
            'worth', 'value', 'money', 'stars', 'review', 'reviews',

            # Common product issues
            'defective', 'issue', 'problem', 'fail', 'failed', 'failure',
            'stop', 'stopped', 'breaking', 'replacement', 'return', 'returned',

            # Size and fit
            'fit', 'fits', 'fitting', 'size', 'sized', 'small', 'large',
            'bigger', 'smaller', 'tight', 'loose',

            # Time-related
            'fast', 'quick', 'slow', 'delay', 'delayed', 'waiting', 'waited',
            'received', 'ordered', 'shipping', 'delivery',

            # Authenticity
            'authentic', 'genuine', 'fake', 'real', 'original', 'counterfeit',
            'legitimate', 'actual'
        ]

        # Initialize base TF-IDF with same parameters as original code
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            stop_words='english',  # Maintains compatibility with your preprocessing
            strip_accents='unicode',
            lowercase=True  # Maintains compatibility with your preprocessing
        )

        self.sentiment_weight = sentiment_weight
        self.sentiment_indices = {}

    def fit(self, X: List[str], y=None):
        # Fit the TF-IDF vectorizer
        self.tfidf.fit(X)

        # Map sentiment words to their indices in the vocabulary
        vocabulary = self.tfidf.vocabulary_
        self.sentiment_indices = {
            word: idx for word in self.sentiment_words
            if word in vocabulary
            for idx in [vocabulary[word]]
        }

        return self

    def transform(self, X: List[str]) -> np.ndarray:
        # Transform using base TF-IDF
        tfidf_matrix = self.tfidf.transform(X)
        weighted_matrix = tfidf_matrix.toarray()

        # Apply sentiment weighting
        for idx in self.sentiment_indices.values():
            weighted_matrix[:, idx] *= self.sentiment_weight

        return weighted_matrix

    def get_feature_names(self) -> List[str]:
        return self.tfidf.get_feature_names_out()

    def get_top_features(self, n: int = 10) -> List[tuple]:
        """Return top n features by their IDF weight."""
        feature_names = self.get_feature_names()
        idf = self.tfidf.idf_
        top_indices = np.argsort(idf)[-n:]
        return [(feature_names[i], idf[i]) for i in top_indices]