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
        self.twoway_maps = {}

    def writeDf(self, fpath: str, sep: str) -> None:
        """
        Write the current dataframe to a text file
        """
        with open(fpath, "w") as f:
            # write the header
            f.write(sep.join([str(col) for col in self.df.columns.tolist()]) + "\n")

            for i, row in self.df.iterrows():
                f.write(sep.join([str(col) for col in row.tolist()]) + "\n")

    def appendDf(self, data_fpath: str, sep: str) -> None:
        """
        Append another dataframe to current one from a text file
        """
        df_to_append = pd.read_csv(data_fpath, sep)
        self.df = self.df.append(df_to_append, ignore_index=True)

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

    def returnNumpyNegSamples(self, k: int, alpha: float, colname: str) -> np.ndarray:
        """
        Assuming [colname] has been indexified, return training data of colname
        in the form of [k] negative samples per row
        """
        counts = self.df.groupby(colname).size().sort_index()
        values = counts.index.values
        
        counts = counts.values.astype(np.float64)
        counts *= alpha
        probs = counts/np.sum(counts)

        return np.random.choice(a=values, size=k*self.df.shape[0], p=probs)

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
            print("{} of {}".format(idx, self.df.shape[0]))
            row[negcolname] = 0
            forbidden_colval = row[colname]
            for _ in range(k):
                negcolval = np.random.choice(a=col_samples, p=probs) 
                while negcolval == forbidden_colval:
                    negcolval = np.random.choice(a=col_samples, p=probs) 

                row[colname] = negcolval
                self.df = self.df.append(row)

    def combineCols(self, col1: str, col2: str, sep: str) -> None:
        """
        Combine two cols to make a new col
        """
        self.df.loc[:, col1+sep+col2] = self.df.loc[:, col1] + sep + self.df.loc[:, col2]

    def removeByNanCol(self, colname: str) -> None:
        "remove rows from self.df where colname's value is nan"
        self.df = self.df.loc[~self.df.loc[:, colname].isna(), :]

    def removeByNumericCol(self, colname: str) -> None:
        """
        Remove rows from self.df where colname's value is numeric
        """
        self.df = self.df.loc[~self.df.loc[:, colname].apply(lambda x: x.isnumeric()), :]

    def createTwoWayMap(self, colname: str) -> None:
        """
        For a given column, assign each unique value an index number from 0 to
        len-1 and save both dict maps
        """
        col_to_idx = {}
        idx_to_col = {}
        for idx, col_val in enumerate(self.df.loc[:, colname].unique().tolist()):
            col_to_idx[col_val] = idx
            idx_to_col[idx] = col_val
            
        self.twoway_maps[colname] = {"col_to_idx":col_to_idx, "idx_to_col":idx_to_col}

    def convertColToIdx(self, colname: str) -> None:
        """
        If a createTwoWayMap has been called on a colname, convert those column
        values to the mapped index
        """
        if colname in self.twoway_maps:
            fnc = lambda row, col_to_idx, colname: col_to_idx[row[colname]]
            kwds = {"func": fnc, 
                    "col_to_idx": self.twoway_maps[colname]["col_to_idx"],
                    "colname": colname}
            self.df.loc[:, colname] = self.df.apply(**kwds, axis=1)

    def convertIdxToCol(self, colname: str) -> None:
        """
        If a createTwoWayMap has been called on a colname, convert those column
        values to the mapped values
        """
        if colname in self.twoway_maps:
            fnc = lambda row, col_to_idx, colname: col_to_idx[row[colname]]
            kwds = {"func": fnc,
                    "col_to_idx": self.twoway_maps[colname]["idx_to_col"],
                    "colname": colname}
            self.df.loc[:, colname] = self.df.apply(**kwds, axis=1)
