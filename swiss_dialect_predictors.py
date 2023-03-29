from abc import ABC

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from byte_pair_tfidf_vectorizer import BytePairTfidfVectorizer


def join_embeddings(series_list: list[pd.Series],
                    arrays: list[np.ndarray],
                    normalize_each_vector=False) -> np.ndarray:
    arrays += [np.stack(s.values) for s in series_list]
    if normalize_each_vector:
        arrays = [a / np.linalg.norm(a, axis=1, keepdims=True) for a in arrays]
    con = np.concatenate(arrays, axis=1)
#    if normalize_each_vector:
#        con = con / np.linalg.norm(con, axis=1, keepdims=True)
    return con


class SwissDialectPredictorInterface(ABC):
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        pass


class SwissDialectPredictorSvmGaussianNB(SwissDialectPredictorInterface):
    def __init__(self,
                 enable_audio=True,
                 enable_sentance_embedding=True,
                 enable_byte_pair_tfidf=True,
                 normalize_each_vector=False,
                 last_classifier=GaussianNB()):
        self.byte_pair_tfidf_vectorizer = BytePairTfidfVectorizer(vocab_size=1000, min_frequency=2)
        self.svm_linear_tfidf = svm.LinearSVC()
        self.gaussian_nb_sentance_embedding_and_audio = last_classifier
        self.enable_audio = enable_audio
        self.enable_sentance_embedding = enable_sentance_embedding
        self.enable_byte_pair_tfidf = enable_byte_pair_tfidf
        self.normalize_each_vector = normalize_each_vector

    def fit(self, df: pd.DataFrame) -> None:
        np_array_list: list[np.ndarray] = []
        series_list: list[pd.Series] = []
        if self.enable_byte_pair_tfidf:
            byte_pair_vectorized = self.byte_pair_tfidf_vectorizer.fit_transform(df["text"].tolist())
            self.svm_linear_tfidf.fit(byte_pair_vectorized, df["label"].tolist())
            np_array_list.append(self.svm_linear_tfidf.decision_function(byte_pair_vectorized))

        if self.enable_sentance_embedding:
            series_list.append(df["sentence_embedding_first_state"])

        if self.enable_audio:
            series_list.append(df["audio"])

        joined_embeddings = join_embeddings(series_list, np_array_list, self.normalize_each_vector)
        self.gaussian_nb_sentance_embedding_and_audio.fit(joined_embeddings, df["label"].tolist())

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        np_array_list: list[np.ndarray] = []
        series_list: list[pd.Series] = []
        if self.enable_byte_pair_tfidf:
            byte_pair_vectorized = self.byte_pair_tfidf_vectorizer.transform(df["text"].tolist())
            np_array_list.append(self.svm_linear_tfidf.decision_function(byte_pair_vectorized))

        if self.enable_sentance_embedding:
            series_list.append(df["sentence_embedding_first_state"])

        if self.enable_audio:
            series_list.append(df["audio"])

        joined_embeddings = join_embeddings(series_list, np_array_list, self.normalize_each_vector)
        return self.gaussian_nb_sentance_embedding_and_audio.predict(joined_embeddings)


class SwissDialectPredictorAllSvmGaussianNB(SwissDialectPredictorInterface):
    def __init__(self,
                 enable_audio=True,
                 enable_sentance_embedding=True,
                 enable_byte_pair_tfidf=True,
                 normalize_each_vector=False,
                 last_classifier=GaussianNB()):
        self.byte_pair_tfidf_vectorizer = BytePairTfidfVectorizer(vocab_size=1000, min_frequency=2)
        self.svm_linear_tfidf = svm.LinearSVC()
        self.gaussian_nb_sentance_embedding_and_audio = last_classifier
        self.enable_audio = enable_audio
        self.enable_sentance_embedding = enable_sentance_embedding
        self.enable_byte_pair_tfidf = enable_byte_pair_tfidf
        self.normalize_each_vector = normalize_each_vector

    def fit(self, df: pd.DataFrame) -> None:
        np_array_list: list[np.ndarray] = []
        series_list: list[pd.Series] = []
        if self.enable_byte_pair_tfidf:
            byte_pair_vectorized = self.byte_pair_tfidf_vectorizer.fit_transform(df["text"].tolist())
            self.svm_linear_tfidf.fit(byte_pair_vectorized, df["label"].tolist())
            np_array_list.append(self.svm_linear_tfidf.decision_function(byte_pair_vectorized))

        if self.enable_sentance_embedding:
            series_list.append(df["sentence_embedding_first_state"])

        if self.enable_audio:
            series_list.append(df["audio"])

        joined_embeddings = join_embeddings(series_list, np_array_list, self.normalize_each_vector)
        self.gaussian_nb_sentance_embedding_and_audio.fit(joined_embeddings, df["label"].tolist())

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        np_array_list: list[np.ndarray] = []
        series_list: list[pd.Series] = []
        if self.enable_byte_pair_tfidf:
            byte_pair_vectorized = self.byte_pair_tfidf_vectorizer.transform(df["text"].tolist())
            np_array_list.append(self.svm_linear_tfidf.decision_function(byte_pair_vectorized))

        if self.enable_sentance_embedding:
            series_list.append(df["sentence_embedding_first_state"])

        if self.enable_audio:
            series_list.append(df["audio"])

        joined_embeddings = join_embeddings(series_list, np_array_list, self.normalize_each_vector)
        return self.gaussian_nb_sentance_embedding_and_audio.predict(joined_embeddings)
