# https://www.spur.uzh.ch/en/departments/research/textgroup/ArchiMob.html

class GdiLoader:
    def __init__(self, set_path="/Users/raphael/Downloads/gdi-vardial-2019/train.txt"):
        self.path = set_path
        self.labels = ["BS", "LU", "ZH", "BE"]

    def get_next(self) -> tuple[str, str]:
        while True:
            with open(self.path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and line[1] in self.labels:
                        text, label = line.split("\t")
                        yield text, label
                    else:
                        raise Exception("Error: ", line)


if __name__ == "__main__":
    loader = GdiLoader(set_path = "/Users/raphael/Downloads/gdi-vardial-2019/train.txt")
    for text, label in loader.get_next():
        print(text, label)