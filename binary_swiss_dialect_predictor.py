import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder

from byte_pair_tfidf_vectorizer import BytePairTfidfVectorizer
from swiss_dialect_predictors import SwissDialectPredictorInterface, join_embeddings


class BinarySwissDialectPredictor(SwissDialectPredictorInterface):
    def __init__(self,
                 classifiers={"audio": GaussianNB(),
                              "sentene_embedding": GaussianNB(),
                              "byte_pair_tfidf": svm.LinearSVC()},
                 enable_audio=True,
                 enable_sentance_embedding=True,
                 enable_byte_pair_tfidf=True,
                 normalize_each_vector=False,
                 last_classifier=GaussianNB(),
                 ):
        self.byte_pair_tfidf_vectorizer = BytePairTfidfVectorizer(vocab_size=1000, min_frequency=2)
        self.tfidf_classifier = classifiers["byte_pair_tfidf"]
        self.audio_classifications = classifiers["audio"]
        self.sentence_embedding_classifications = classifiers["sentene_embedding"]
        self.enable_audio = enable_audio
        self.enable_sentance_embedding = enable_sentance_embedding
        self.enable_byte_pair_tfidf = enable_byte_pair_tfidf
        self.normalize_each_vector = normalize_each_vector
        self.last_classifier = last_classifier
        self.one_hot_encoder = OneHotEncoder(sparse=False)
        self.one_hot_encoder.fit(np.array(["LU", "BE", "BS", "ZH"]).reshape(-1, 1))

    def fit(self, df: pd.DataFrame) -> None:
        first_step_prediction_list: list[np.ndarray] = []
        if self.enable_byte_pair_tfidf:
            byte_pair_vectorized = self.byte_pair_tfidf_vectorizer.fit_transform(df["text"].tolist())
            self.tfidf_classifier.fit(byte_pair_vectorized, df["label"].tolist())
            prediction_byte_pair_tfidf = self.tfidf_classifier.predict(byte_pair_vectorized)
            one_hot_encoded = self.one_hot_encoder.transform(prediction_byte_pair_tfidf.reshape(-1, 1))
            first_step_prediction_list.append(one_hot_encoded)

        if self.enable_sentance_embedding:
            x_sentence_embedding = df["sentence_embedding_first_state"].tolist()
            self.sentence_embedding_classifications.fit(x_sentence_embedding, df["label"].tolist())
            prediction_sentence_embedding = self.sentence_embedding_classifications.predict(x_sentence_embedding)
            one_hot_encoded = self.one_hot_encoder.transform(prediction_sentence_embedding.reshape(-1, 1))
            first_step_prediction_list.append(one_hot_encoded)

        if self.enable_audio:
            x_audio = df["audio"].tolist()
            self.audio_classifications.fit(x_audio, df["label"].tolist())
            prediction_audio = self.audio_classifications.predict(x_audio)
            one_hot_encoded = self.one_hot_encoder.transform(prediction_audio.reshape(-1, 1))
            first_step_prediction_list.append(one_hot_encoded)

        joined_embeddings = join_embeddings([], first_step_prediction_list, self.normalize_each_vector)
        self.last_classifier.fit(joined_embeddings, df["label"].tolist())

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        first_step_prediction_list: list[np.ndarray] = []
        if self.enable_byte_pair_tfidf:
            byte_pair_vectorized = self.byte_pair_tfidf_vectorizer.transform(df["text"].tolist())
            prediction_byte_pair_tfidf = self.tfidf_classifier.predict(byte_pair_vectorized)
            one_hot_encoded = self.one_hot_encoder.transform(prediction_byte_pair_tfidf.reshape(-1, 1))
            first_step_prediction_list.append(one_hot_encoded)

        if self.enable_sentance_embedding:
            x_sentence_embedding = df["sentence_embedding_first_state"].tolist()
            prediction_sentence_embedding = self.sentence_embedding_classifications.predict(x_sentence_embedding)
            one_hot_encoded = self.one_hot_encoder.transform(prediction_sentence_embedding.reshape(-1, 1))
            first_step_prediction_list.append(one_hot_encoded)

        if self.enable_audio:
            x_audio = df["audio"].tolist()
            prediction_audio = self.audio_classifications.predict(x_audio)
            one_hot_encoded = self.one_hot_encoder.transform(prediction_audio.reshape(-1, 1))
            first_step_prediction_list.append(one_hot_encoded)

        joined_embeddings = join_embeddings([], first_step_prediction_list, self.normalize_each_vector)
        return self.last_classifier.predict(joined_embeddings)
