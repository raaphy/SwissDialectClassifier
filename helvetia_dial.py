import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

from byte_pair_tfidf_vectorizer import BytePairTfidfVectorizer
from experimental_predictors import SwissDialectPredictorInterface


class HelvetiaDial(SwissDialectPredictorInterface):
    def __init__(self,
                 enable_sentence_embedding=True,
                 enable_byte_pair_tfidf=True,
                 enable_audio=False):
        self.byte_pair_tfidf_vectorizer = BytePairTfidfVectorizer(vocab_size=1000, min_frequency=2)
        self.tfidf_classifier = svm.LinearSVC(C=1.0) if enable_byte_pair_tfidf else None
        self.audio_classifier = GaussianNB() if enable_audio else None
        self.sentence_embedding_classifications = GaussianNB() if enable_sentence_embedding else None
        self.last_classifier = GaussianNB()

    def fit(self, df: pd.DataFrame) -> None:
        first_stage_outputs: list[np.ndarray] = []
        y = df["label"].tolist()
        if self.tfidf_classifier is not None:
            byte_pair_vectorized = self.byte_pair_tfidf_vectorizer.fit_transform(df["text"].tolist())
            self.tfidf_classifier.fit(byte_pair_vectorized, y)
            first_stage_outputs.append(self.tfidf_classifier.decision_function(byte_pair_vectorized))

        if self.sentence_embedding_classifications is not None:
            x_sentence_embedding = df["sentence_embedding_first_state"].tolist()
            self.sentence_embedding_classifications.fit(x_sentence_embedding, y)
            prediction_sentence_embedding = self.sentence_embedding_classifications.predict_proba(x_sentence_embedding)
            first_stage_outputs.append(prediction_sentence_embedding)

        if self.audio_classifier is not None:
            x_audio = df["audio"].tolist()
            self.audio_classifier.fit(x_audio, y)
            prediction_audio = self.audio_classifier.predict_proba(x_audio)
            first_stage_outputs.append(prediction_audio)
        joined_embeddings = np.concatenate(first_stage_outputs, axis=1)
        self.last_classifier.fit(joined_embeddings, y)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        first_stage_outputs: list[np.ndarray] = []
        if self.tfidf_classifier is not None:
            byte_pair_vectorized = self.byte_pair_tfidf_vectorizer.transform(df["text"].tolist())
            first_stage_outputs.append(self.tfidf_classifier.decision_function(byte_pair_vectorized))

        if self.sentence_embedding_classifications is not None:
            x_sentence_embedding = df["sentence_embedding_first_state"].tolist()
            prediction_sentence_embedding = self.sentence_embedding_classifications.predict_proba(x_sentence_embedding)
            first_stage_outputs.append(prediction_sentence_embedding)

        if self.audio_classifier is not None:
            x_audio = df["audio"].tolist()
            prediction_audio = self.audio_classifier.predict_proba(x_audio)
            first_stage_outputs.append(prediction_audio)

        joined_embeddings = np.concatenate(first_stage_outputs, axis=1)
        return self.last_classifier.predict(joined_embeddings)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

    # Load the data
    train_df = pd.read_feather("data/train_embedding_bert_swiss_lm.feather")
    dev_df = pd.read_feather("data/dev_embedding_bert_swiss_lm.feather")
    test_df = pd.read_feather("data/test_embedding_bert_swiss_lm.feather")

    # Train the model on the training data
    predictor_trained_on_train = HelvetiaDial()
    predictor_trained_on_train.fit(train_df)

    # Evaluate the model on the train data
    y_pred = predictor_trained_on_train.predict(train_df)
    print(f"Trained on train_df. Train F1-score is:\t\t\t\t"
          "{:.4f}".format(f1_score(train_df['label'].tolist(), y_pred, average='macro')))
    # print(classification_report(train_df["label"].tolist(), y_pred))

    # Evaluate the same model on the dev data
    y_pred = predictor_trained_on_train.predict(dev_df)
    print(f"Trained on train_df. Validation F1-score is:\t\t"
          "{:.4f}".format(f1_score(dev_df['label'].tolist(), y_pred, average='macro')))
    # print(classification_report(dev_df["label"].tolist(), y_pred_train))

    # Train the model on the train and dev data
    predictor_trained_on_train_and_dev = HelvetiaDial()
    predictor_trained_on_train_and_dev.fit(pd.concat([train_df, dev_df]))
    # Evaluate the model on the test data
    y_pred = predictor_trained_on_train_and_dev.predict(test_df)
    print(f"Trained on train_df and dev_df. Test F1-score is:\t"
          "{:.4f}".format(f1_score(test_df['label'].tolist(), y_pred, average='macro')))
    # print(classification_report(test_df["label"].tolist(), y_pred))

    # Plot the confusion matrix
    print("Plot the confusion matrix for test data")
    cm = confusion_matrix(test_df["label"].tolist(), y_pred, labels=["ZH", "LU", "BE", "BS"])
    ConfusionMatrixDisplay(cm, display_labels=["Zurich", "Lucerne", "Bern", "Basel"]).plot()
    plt.title("Confusion matrix for test data")
    plt.show()
