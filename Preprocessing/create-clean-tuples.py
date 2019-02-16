from itertools import combinations
import multiprocessing as mp
import pickle
import string
import re

def createCleanTuples(line, queue, translator, countryforms):
    date, text = line.split(",")
    
    # Get rid of punctuation and gaurentee lowercase
    text = text.translate(translator).lower()

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
            [(entities[i], contexts[i], contexts[i+1]) for i in range(len(entities))], 2):
            queue.put("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(date, c1[0], c2[0], c1[1], c1[2], c2[1], c2[2]))

def listener(queue, header):
    with open("../Data/abcnews-clean.txt", "w") as clean_data:
        # Write header
        clean_data.write(header)

        while True:
            msg = queue.get()
            if msg == "KILL":
                break
            clean_data.write(msg)
            clean_data.flush()

if __name__ == "__main__":
    # Load headlines from abc, split them up
    text_file = open("../Data/abcnews-date-text.csv")
    lines = [line for line in text_file.read().splitlines()]

    # Load proper countryforms to replace the forms in headlines
    with open("../Data/countryforms.pickle", "rb") as handle:
        countryforms = pickle.load(handle)

    # Create translator for filtering out punctuation
    translator = str.maketrans("", "", string.punctuation)

    # Start manager, queue and pool
    manager = mp.Manager()
    queue = manager.Queue()
    pool = mp.Pool(8)

    # Start listener
    header = "date\tc1\tc2\tc1_ctxt_l\tc1_ctxt_r\tc2_ctxt_l\tc2_ctxt_r\n"
    watcher = pool.apply_async(listener, (queue, header,))

    results = []
    for i, line in enumerate(lines[1:]):
        print("{} of {}".format(i, len(lines)))
        result = pool.apply_async(createCleanTuples, (line, queue, translator, countryforms))
        results.append(result)

    # Kill queue once all is done
    [result.wait() for result in results]
    queue.put("KILL")
    pool.close()
