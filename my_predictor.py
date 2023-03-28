import random

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report


class MyPredictor:
    def __init__(self,
                 path_train_embeddings,
                 path_dev_embeddings,
                 path_test_embeddings=None,
                 which_embedding=['sentence_embedding_first_state'],
                 audio_length=20):
        self.clf = None
        self.path_train_embeddings = path_train_embeddings
        self.path_dev_embeddings = path_dev_embeddings
        self.path_test_embeddings = path_test_embeddings
        self.which_embedding = which_embedding
        self.audio_indixes = random.sample(range(400), audio_length)

    def fit(self, kernel='linear', C=1.0):
        training_embeddings = pd.read_feather(self.path_train_embeddings)
        self.clf = svm.SVC(C=C, kernel=kernel)
        embedding = self.generate_embedding(training_embeddings)
        self.clf.fit(embedding, training_embeddings["label"].tolist())

    def predict_all(self):
        training_embeddings = pd.read_feather(self.path_train_embeddings)
        train_prediction = self.clf.predict(self.generate_embedding(training_embeddings))
        print("Training set:")
        print(classification_report(training_embeddings["label"].tolist(), train_prediction))

        dev_embeddings = pd.read_feather(self.path_dev_embeddings)
        dev_prediction = self.clf.predict(self.generate_embedding(dev_embeddings))
        print("Dev set:")
        print(classification_report(dev_embeddings["label"].tolist(), dev_prediction))

        if self.path_test_embeddings is not None:
            test_embeddings = pd.read_feather(self.path_test_embeddings)
            test_prediction = self.clf.predict(self.generate_embedding(test_embeddings).tolist())
            print("Test set:")
            print(classification_report(test_embeddings["label"].tolist(), test_prediction))

    def generate_embedding(self, training_embeddings):
        training_embeddings = training_embeddings.copy()
        training_embeddings["audio"] = training_embeddings["audio"].apply(lambda x: x[self.audio_indixes])
        con = np.concatenate([np.stack(training_embeddings[col].values) for col in self.which_embedding], axis=1)

        # concatenated = np.concatenate([training_embeddings[col].to_numpy() for col in self.which_embedding], axis=1)
        # change name of column
        return con