import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
# https://github.com/jungomi/swiss-language-model

class BertSwissLm:

    def __init__(self, path="/Users/raphael/Downloads/bert-swiss-lm"):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path)

    def get_sentence_embedding(self, text) -> tuple[np.ndarray, np.ndarray]:
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state[0, 0, :]
            pooler_output = outputs.pooler_output[0, :]
        return last_hidden_state.numpy(), pooler_output.numpy()
   #     sentence_embedding = torch.mean(last_hidden_state, dim=1)
#        print(last_hidden_state)
#        print('PyCharm')

    def add_sentence_embedding(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[["sentence_embedding_first_state", "sentence_embedding_pooler"]] = df["text"].apply(self.get_sentence_embedding).apply(
            pd.Series)
        #df[["sentence_embedding", "sentence_embedding_pooler"]] = df["text"].apply(self.get_sentence_embedding)
        return df

    def add_sentence_embedding_fast(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # tokanize all df["text"] at once
        inputs = self.tokenizer(df["text"].tolist(), padding=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state[:, 0, :]
            pooler_output = outputs.pooler_output[:, :]
        df["sentence_embedding_first_state"] = last_hidden_state.numpy().tolist()
        df["sentence_embedding_pooler_output"] = pooler_output.numpy().tolist()
        return df

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
