from collections import Counter
import numpy as np
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
            f.write(sep.join([str(col) for col in self.df.columns.tolist()]) + "\n")

            for i, row in self.df.iterrows():
                f.write(sep.join([str(col) for col in row.tolist()]) + "\n")

    def stackSepCol(self, colname: str, sep: str, newcolname: str) -> None:
        """
        Stack the df by a certain column that is seperated by some string
        """
        df = pd.DataFrame()
        for idx, row in self.df.iterrows():
            if pd.isna(row[colname]):
                continue

            for el in row[colname].split(sep):
                df = df.append(row.append(pd.Series({newcolname:el})), ignore_index=True)

        df.drop(labels=colname, axis="columns", inplace=True)
        self.df = df

    def subsample(self, t: float, subcolname: str) -> None:
        """
        Subsample based on col frequency frequency
        """
        # Get frequencies
        freqs = Counter()
        for idx, row in self.df.iterrows():
            freqs[row[subcolname]] += 1/self.df.shape[0]

        # Do subsampling
        rows_to_drop = []
        for idx, row in self.df.iterrows():
            f = freqs[row[subcolname]]
            if np.random.rand() < ((f-t)/f)-(t/f)**0.5:
                rows_to_drop.append(idx)

        self.df.drop(labels=rows_to_drop, axis="index", inplace=True)

    def generateNegSamples(self, k: int, alpha: float, colname: str, 
                           negcolname: str) -> None:
        """
        Append negative samples to self.df, k amount per row
        """
        # Get counts
        counts = Counter()
        for idx, row in self.df.iterrows():
            counts[row[colname]] += 1

        # Take all counts to the alpha 
        for key, count in counts.items():
            counts[key] = count**alpha

        # Convert counts to frequencies
        freqs = {}
        denom = sum([count for count in counts.values()])
        for key, count in counts.items():
            freqs[key] = count/denom

        # Generate neg samples
        col_samples = [key for key in freqs.keys()]
        probs = [prob for prob in freqs.values()]
        for idx, row in self.df.iterrows():
            row[negcolname] = 0
            forbidden_colval = row[colname]
            for _ in range(k):
                negcolval = np.random.choice(a=col_samples, p=probs) 
                while negcolval == forbidden_colval:
                    negcolval = np.random.choice(a=col_samples, p=probs) 

                row[colname] = negcolval
                self.df = self.df.append(row)
             
if __name__ == "__main__":
    fcp = FullContextProcessor("../Data/ABC-News/abcnews-complete-ctxt.txt", "\t")
    fcp.df.loc[:, "pos"] = 1
    fcp.stackSepCol(colname="ctxt", sep=",", newcolname="word")
    fcp.subsample(t=1e-5, subcolname="word")
    fcp.generateNegSamples(k=5, alpha=0.75, colname="word", negcolname="pos")
    fcp.writeDf("../Data/ABC-News/abcnews-sgns-processed.txt", "\t")
