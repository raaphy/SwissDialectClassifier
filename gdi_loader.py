# https://www.spur.uzh.ch/en/departments/research/textgroup/ArchiMob.html
import numpy as np
import pandas as pd

class GdiLoader:
    def __init__(self, set_path="/Users/raphael/Downloads/gdi-vardial-2019/"):
        self.path = set_path
        self.labels = ["BS", "LU", "ZH", "BE"]

    def _get_next(self) -> tuple[str, str]:
        while True:
            with open(self.path, "r") as f:
                for line in f:
                    line = line.strip()
                    try:
                        text, label = line.split("\t")
                    except:
                        raise Exception("Error: ", line)
                    if label in self.labels:
                        yield text, label
                    else:
                        raise Exception("Error: ", line)

    def row_to_np_array(self, row: pd.Series):
        return np.array(row.values)

    def create_dataframe(self, which_type="train") -> pd.DataFrame:
        if which_type == "train" or which_type == "dev":
            df = pd.read_csv(self.path + which_type + ".txt", delimiter='\t', names=["text", "label"])
            df_audio = pd.read_csv(self.path + which_type + ".vec", sep=' ', header=None)
            df["audio"] = df_audio.apply(self.row_to_np_array, axis=1)
            df.to_feather(self.path + which_type + ".feather")
        elif which_type == "test":
            text_df = pd.read_csv(self.path+ "test.txt", delimiter='\t', names=["text"])
            labels_df = pd.read_csv(self.path + ".labels", delimiter='\t', names=["label"])
            df = pd.concat([text_df, labels_df], axis=1)
            df.to_feather(self.path+  "test.feather")
        else:
            raise Exception("Error: ", which_type)
        return df

    #def create_dataframe_test(self) -> pd.DataFrame:
        # text is in self.path and labels in self.path + ".labels"


if __name__ == "__main__":
    loader = GdiLoader(set_path = "/Users/raphael/Downloads/gdi-vardial-2019/")
    loader.create_dataframe(which_type="train")
    loader.create_dataframe(which_type="dev")
    #for text, label in loader.get_next():
     #   print(text, label)