import pandas as pd

class FullContextProcessor:
    """
    Object for processing full context extracted data with format:
        date (yyyymmdd) \t c1 (XYZ) \t c2 (XYZ) \t ctxt (w1,w2,...,wn) 
    """
    def __init__(self, data_fpath: str, sep: str):
        self.df = pd.read_csv(data_fpath, sep)

    def writeDf(self, fpath: str, sep: str) -> None:
        with open(fpath, "w") as f:
            # write the header
            f.write(sep.join(df.columns.tolist()) + "\n")

            for i, row in self.df.iterrows():
                print(i) 
    
if __name__ == "__main__":
    import ipdb; ipdb.set_trace()
    df = pd.read_csv("../Data/ABC-News/abcnews-complete-ctxt.txt", sep="\t") 
