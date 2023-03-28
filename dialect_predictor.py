
class DialectPredictor:
    def __init__(self,
                 bert_model_path="/Users/raphael/Downloads/bert-swiss-lm",
                 gdi_path="/Users/raphael/Downloads/gdi-vardial-2019/",
                 cache="/Users/raphael/Downloads/swiss_dialects_cache/"):
        self.bert_model = BertSwissLm(path=bert_model_path)
        self.gdi_loader = GdiLoader(set_path=gdi_path, cache=cache)
        self.cache = cache

    def fit(self, which_type="train"):
        df = self.gdi_loader.get_data(which_type=which_type)
        df["sentence_embedding"] = df["text"].apply(self.bert_model.get_sentence_embedding)
        return df
