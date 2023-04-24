import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


class BertSwissLm:
    def __init__(self, path="/Users/raphael/Downloads/bert-swiss-lm", set_default_language_to_de_CH=False):
        """
        :param path: Path to the model, either a local path or a huggingface model name
        :param set_default_language_to_de_CH: If True, the model will be set to the default language "de_CH"
            this is needed for SwissBert.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path)
        if set_default_language_to_de_CH:
            self.model.set_default_language("de_CH")

    def _get_sentence_embedding(self, text) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state[0, 0, :]
            #pooler_output = outputs.pooler_output[0, :]
        return last_hidden_state.numpy()

    def add_sentence_embedding_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["sentence_embedding_first_state"] = df["text"].apply(
            self._get_sentence_embedding).apply(pd.Series)
        return df

