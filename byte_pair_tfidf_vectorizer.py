import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizers.implementations import ByteLevelBPETokenizer


class BytePairTfidfVectorizer:
    def __init__(self, vocab_size: int, min_frequency: int):
        self._vectorizer = None
        self._tokenizer = None
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency

    def fit_transform(self, x: list[str]) -> np.ndarray:
        self._tokenizer = ByteLevelBPETokenizer()
        self._tokenizer.train_from_iterator(iter(x), vocab_size=self.vocab_size, min_frequency=self.min_frequency,
                                            show_progress=False)
        encoded = self._tokenizer.encode_batch(x)
        tokenized = [e.tokens for e in encoded]
        self._vectorizer = TfidfVectorizer()
        tfidf_weights = self._vectorizer.fit_transform([' '.join(tokens) for tokens in tokenized]).toarray()
        return tfidf_weights

    def transform(self, x: list[str]) -> np.ndarray:
        encoded = self._tokenizer.encode_batch(x)
        tokenized = [e.tokens for e in encoded]
        tfidf_weights = self._vectorizer.transform([' '.join(tokens) for tokens in tokenized]).toarray()
        return tfidf_weights
