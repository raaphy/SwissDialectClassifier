import numpy as np
import pandas as pd

from abc import ABC
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

from byte_pair_tfidf_vectorizer import BytePairTfidfVectorizer


def join_embeddings(series_list: list[pd.Series],
                    arrays: list[np.ndarray],
                    normalize_each_vector=False) -> np.ndarray:
    arrays += [np.stack(s.values) for s in series_list]
    if normalize_each_vector:
        arrays = [a / np.linalg.norm(a, axis=1, keepdims=True) for a in arrays]
    con = np.concatenate(arrays, axis=1)
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


def _sort_probabilities(feature_names: np.ndarray, prediction_sentence_embedding: np.ndarray) -> np.ndarray:
    label_order = ["LU", "ZH", "BS", "BE"]
    # in which row is "LU" in the feature_names (which is a numpy string array)
    label_ids = [np.where(feature_names == label_order[i])[0][0] for i in range(4)]

    prediction_sentence_embedding = prediction_sentence_embedding[:, [label_ids[0],
                                                                      label_ids[1],
                                                                      label_ids[2],
                                                                      label_ids[3]]]
    return prediction_sentence_embedding


class SwissDialectPredictorSeperateGaussians(SwissDialectPredictorInterface):
    def __init__(self,
                 audio_classifier=GaussianNB(),
                 sentence_embedding_classifier=GaussianNB(),
                 tfidf_classifier=svm.LinearSVC(),
                 enable_audio=True,
                 enable_sentance_embedding=True,
                 enable_byte_pair_tfidf=True,
                 normalize_each_vector=False,
                 last_classifier=GaussianNB(),
                 audio_weight=0.5,
                 ):
        self.byte_pair_tfidf_vectorizer = BytePairTfidfVectorizer(vocab_size=1000, min_frequency=2)
        self.tfidf_classifier = tfidf_classifier
        self.audio_classifier = audio_classifier
        self.sentence_embedding_classifications = sentence_embedding_classifier
        self.enable_audio = enable_audio
        self.enable_sentance_embedding = enable_sentance_embedding
        self.enable_byte_pair_tfidf = enable_byte_pair_tfidf
        self.normalize_each_vector = normalize_each_vector
        self.last_classifier = last_classifier
        self.audio_weight = audio_weight

    def fit(self, df: pd.DataFrame) -> None:
        np_array_list: list[np.ndarray] = []
        probability_list: list[np.ndarray] = []
        if self.enable_byte_pair_tfidf:
            byte_pair_vectorized = self.byte_pair_tfidf_vectorizer.fit_transform(df["text"].tolist())
            self.tfidf_classifier.fit(byte_pair_vectorized, df["label"].tolist())
            np_array_list.append(self.tfidf_classifier.decision_function(byte_pair_vectorized))

        if self.enable_sentance_embedding:
            x_sentence_embedding = df["sentence_embedding_first_state"].tolist()
            self.sentence_embedding_classifications.fit(x_sentence_embedding, df["label"].tolist())
            prediction_sentence_embedding = self.sentence_embedding_classifications.predict_proba(x_sentence_embedding)
            feature_names = self.sentence_embedding_classifications.classes_
            # reorder prediction to match the order of the labels
            prediction_sentence_embedding = _sort_probabilities(feature_names, prediction_sentence_embedding)
            probability_list.append(prediction_sentence_embedding)

        if self.enable_audio:
            x_audio = df["audio"].tolist()
            self.audio_classifier.fit(x_audio, df["label"].tolist())
            prediction_audio = self.audio_classifier.predict_proba(x_audio)
            feature_names = self.audio_classifier.classes_
            prediction_audio = _sort_probabilities(feature_names, prediction_audio)
            prediction_audio = prediction_audio ** self.audio_weight
            probability_list.append(prediction_audio)
        if len(probability_list) > 0:
            average_of_probabilities = np.product(probability_list, axis=0)
            np_array_list.append(average_of_probabilities)
            joined_embeddings = join_embeddings([], np_array_list, self.normalize_each_vector)
            self.last_classifier.fit(joined_embeddings, df["label"].tolist())
        else:
            joined_embeddings = join_embeddings([], np_array_list, self.normalize_each_vector)
            self.last_classifier.fit(joined_embeddings, df["label"].tolist())

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        np_array_list: list[np.ndarray] = []
        probability_list: list[np.ndarray] = []
        if self.enable_byte_pair_tfidf:
            byte_pair_vectorized = self.byte_pair_tfidf_vectorizer.transform(df["text"].tolist())
            np_array_list.append(self.tfidf_classifier.decision_function(byte_pair_vectorized))

        if self.enable_sentance_embedding:
            x_sentence_embedding = df["sentence_embedding_first_state"].tolist()
            prediction_sentence_embedding = self.sentence_embedding_classifications.predict_proba(x_sentence_embedding)
            feature_names = self.sentence_embedding_classifications.classes_
            # reorder prediction to match the order of the labels
            prediction_sentence_embedding = _sort_probabilities(feature_names, prediction_sentence_embedding)
            probability_list.append(prediction_sentence_embedding)

        if self.enable_audio:
            x_audio = df["audio"].tolist()
            prediction_audio = self.audio_classifier.predict_proba(x_audio)
            feature_names = self.audio_classifier.classes_
            prediction_audio = _sort_probabilities(feature_names, prediction_audio)
            prediction_audio = prediction_audio ** self.audio_weight
            probability_list.append(prediction_audio)

        if len(probability_list) > 0:
            average_of_probabilities = np.product(probability_list, axis=0)
            np_array_list.append(average_of_probabilities)
            joined_embeddings = join_embeddings([], np_array_list, self.normalize_each_vector)
            return self.last_classifier.predict(joined_embeddings)
        else:
            joined_embeddings = join_embeddings([], np_array_list, self.normalize_each_vector)
            return self.last_classifier.predict(joined_embeddings)
