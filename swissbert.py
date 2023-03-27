import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
# https://huggingface.co/ZurichNLP/swissbert

# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("ZurichNLP/swissbert")
    model = AutoModel.from_pretrained("ZurichNLP/swissbert")
 #   model = AutoModelForMaskedLM.from_pretrained("ZurichNLP/swissbert")
    model.set_default_language("de_CH")
    text = ["Ich bin ein Berliner."] * 1000
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state[:, 0, :]
        #last_hidden_state = outputs.pooler_output[:, 0, :]

   # sentence_embedding = torch.mean(last_hidden_state, dim=1)
    print(last_hidden_state)
    print_hi('PyCharm')


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
