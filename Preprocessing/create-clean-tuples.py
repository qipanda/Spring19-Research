import ipdb; ipdb.set_trace()
text_file = open("../Data/abcnews-date-text.csv")
lines = text_file.readlines()

curmax = 0
for idx, line in enumerate(lines[1:]):
    headline_id, text = line.split(",")
    curmax = max(curmax, len(text.split(" ")))
    print(idx)

print(curmax)

