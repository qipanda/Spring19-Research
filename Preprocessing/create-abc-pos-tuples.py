import pandas as pd

if __name__ == "__main__":
    # Get abv complete data
    data = pd.read_csv("../Data/ABC-News/abcnews-complete-ctxt.txt", sep="\t")

    with open("../Data/ABC-News/abcnews-complete-pos.txt", "w") as f:
        # write header
        f.write("date\tc1\tc2\tword\tpos\n")

        for i, row in data.iterrows():
            print(i)
            if pd.isna(row["ctxt"]):
                continue

            for word in row["ctxt"].split(","):
                if not pd.isna(word) and len(word) > 0:
                    f.write("{}\t{}\t{}\t{}\t{}\n".format(
                        row["date"], row["c1"], row["c2"], word, True))

