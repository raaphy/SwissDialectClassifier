import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
# https://github.com/jungomi/swiss-language-model

# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Pfad zum Ordner mit den Modell-Dateien
    model_path = "/Users/raphael/Downloads/bert-swiss-lm"

    # Laden des Tokenizers
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Laden des Modells
    model = AutoModel.from_pretrained(model_path)

    text = ["Ich bin ein Berliner."] * 1000
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state[:, 0, :]
        #last_hidden_state = outputs.pooler_output[:, 0, :]

   # sentence_embedding = torch.mean(last_hidden_state, dim=1)
    print(last_hidden_state)
    print('PyCharm')


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
