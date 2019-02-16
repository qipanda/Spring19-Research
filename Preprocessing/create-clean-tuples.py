from itertools import combinations
import pickle
import string
import re

# import ipdb; ipdb.set_trace()
# with open("../Data/clean-country-pair-contexts.pickle", "rb") as handle:
#     test = pickle.load(handle)

# Load countryforms to replace the forms in headlines
with open("../Data/countryforms.pickle", "rb") as handle:
    countryforms = pickle.load(handle)

# Load headlines from abc
text_file = open("../Data/abcnews-date-text.csv")
lines = text_file.readlines()

# Create translator for filtering out punctuation
translator = str.maketrans("", "", string.punctuation)

# Process each line into proper tuple (ignore header row)
clean_tuples = []
for i, line in enumerate(lines[1:]):
    print("{} of {}".format(i, len(lines[1:])))
    date, text = line.split(",")
    
    # Get rid of '\n' char in text and get rid of punctuation
    text = text.strip("\n").translate(translator)

    # Replace string forms of countries with standard form
    for key, val in countryforms.items():
        text = re.sub(key, val, text) 

    # Split the string by spaces to get the tokens
    text = text.split(" ")

    # Find all entities and contexts
    entities = []
    contexts = []
    context_buffer = ""
    for i, token in enumerate(text):
        if token.isupper():
            entities.append(token)
            contexts.append(context_buffer.strip(","))
            context_buffer = ""
        else:
            context_buffer += token + ","
    contexts.append(context_buffer.strip(","))

    # If at least two entities, append them to clean data with l and r contexts
    if len(entities) > 1:
        for c1, c2 in combinations(
            [(date, entities[i], contexts[i], contexts[i+1]) for i in range(len(entities))], 2):
            clean_tuples.append((c1, c2))

with open("../Data/clean-country-pair-contexts.pickle", "wb") as handle:
    pickle.dump(clean_tuples, handle)
