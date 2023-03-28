import pandas as pd
from sklearn import svm
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report


class MyPredictor:
    def __init__(self, path_train_embeddings, path_dev_embeddings, path_test_embeddings=None, which_embedding=["sentence_embedding_first_state"]):
        self.clf = None
        self.path_train_embeddings = path_train_embeddings
        self.path_dev_embeddings = path_dev_embeddings
        self.path_test_embeddings = path_test_embeddings
        self.which_embedding = which_embedding

    def fit(self ):
        training_embeddings = pd.read_feather(self.path_train_embeddings)
        self.clf = svm.SVC()
        #embedding = self.generate_embedding(training_embeddings)
        self.clf.fit(training_embeddings[self.which_embedding].tolist(), training_embeddings["label"].tolist())

    def predict_all(self):
        training_embeddings = pd.read_feather(self.path_train_embeddings)
        train_prediction = self.clf.predict(training_embeddings[self.which_embedding].tolist())
        print("Training set:")
        print(classification_report(training_embeddings["label"].tolist(), train_prediction))

        dev_embeddings = pd.read_feather(self.path_dev_embeddings)
        dev_prediction = self.clf.predict(dev_embeddings[self.which_embedding].tolist())
        print("Dev set:")
        print(classification_report(dev_embeddings["label"].tolist(), dev_prediction))

        if self.path_test_embeddings is not None:
            test_embeddings = pd.read_feather(self.path_test_embeddings)
            test_prediction = self.clf.predict(test_embeddings[self.which_embedding].tolist())
            print("Test set:")
            print(classification_report(test_embeddings["label"].tolist(), test_prediction))

    def generate_embedding(self, training_embeddings):
        concatenated = pd.concat([training_embeddings[col] for col in self.which_embedding], axis=1)
        return concatenated