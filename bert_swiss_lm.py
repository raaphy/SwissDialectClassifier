import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
# https://github.com/jungomi/swiss-language-model

class BertSwissLm:

    def __init__(self, path="/Users/raphael/Downloads/bert-swiss-lm"):

        # Laden des Tokenizers
        self.tokenizer = AutoTokenizer.from_pretrained(path)

        # Laden des Modells
        self.model = AutoModel.from_pretrained(path)

#        text = ["Ich bin ein Berliner."] * 1000

    def get_sentence_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state[:, 0, :]
            #last_hidden_state = outputs.pooler_output[:, 0, :]
        return last_hidden_state
   #     sentence_embedding = torch.mean(last_hidden_state, dim=1)
#        print(last_hidden_state)
#        print('PyCharm')


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
