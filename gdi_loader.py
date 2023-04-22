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
                    except ValueError:
                        raise Exception("Error: ", line)
                    if label in self.labels:
                        yield text, label
                    else:
                        raise Exception("Error: ", line)

    @staticmethod
    def _row_to_np_array(row: pd.Series):
        return np.array(row.values)

    def create_dataframe(self, which_type="train") -> pd.DataFrame:
        df = pd.read_csv(self.path + which_type + ".txt", delimiter='\t', names=["text", "label"])
        df_audio = pd.read_csv(self.path + which_type + ".vec", sep=' ', header=None)
        df["audio"] = df_audio.apply(self._row_to_np_array, axis=1)
        df.to_feather(self.path + which_type + ".feather")
        return df


if __name__ == "__main__":
    loader = GdiLoader(set_path="/Users/raphael/Downloads/gdi-vardial-2019/")
    train_df = loader.create_dataframe(which_type="train")
    dev_df = loader.create_dataframe(which_type="dev")
