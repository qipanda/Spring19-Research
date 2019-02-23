from typing import Dict, Tuple, Optional
from itertools import combinations
import multiprocessing as mp

def writeWindowedTuple(idx: int, line: Tuple[str, str], queue: mp.Queue, 
                       translator: Dict[int, Optional[int]], 
                       countryforms: Dict[str, str]) -> None:
    """
    Creates tuples from a given headline containing the words near each country
    but not past another country. 

    Example, if the text is:
        [ctxt_1] [country_1] [ctxt_2] [country_2] [ctxt_3] [country_3] [ctxt_4]

    Then the following 3C2=3 tuples will be created:
        country_1, country_2, with ctxt_[1,2,2,3]
        country_1, country_3, with ctxt_[1,2,3,4]
        country_2, country_3, with ctxt_[2,3,3,4]

    INPUT:
        idx:        integer for keeping track of preprocessing progress
        line:       tuple containing (date string, text of headline) 
        queue:      object to handle writing to output file
        translator:     dictionary of ascii to substituion ascii mappings
        countryforms:   dict mapping textforms to proper country code   
    OUPUT:
        None directly, but writes tuples to textfile based on queue
    """
    # Seperate the tuple
    date, headline = line

    # Lowercase, remove based on trans, trim sides, and split by space for headline
    split_headline = headline.lower().translate(translator).strip().split(" ")

    # Find all the entities and their context windows
    entities = []
    contexts = []
    context_buffer = ""
    for i, token in enumerate(split_headline):
        if token in countryforms:
            entities.append(countryforms[token])
            contexts.append(context_buffer.strip(","))
            context_buffer = ""
        else:
            context_buffer += token + ","
    contexts.append(context_buffer.strip(","))
        
    # If at least two entities, append them to clean data with l and r contexts
    if len(entities) > 1:
        for c1, c2 in combinations(
            [(entities[i], contexts[i], contexts[i+1]) for i in range(len(entities))], 2):
            # Ensure alphabetical c1, c2 ordering
            if c1[0] > c2[0]:
                c1, c2 = c2, c1

            # Add the tuple to be written on the queue
            queue.put((idx,"{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                date, c1[0], c2[0], c1[1], c1[2], c2[1], c2[2])))

def writeCompleteTuple(idx: int, line: Tuple[str, str], queue: mp.Queue, 
                       translator: Dict[int, Optional[int]],
                       countryforms: Dict[str, str]) -> None:
    """
    Creates tuples from a given headline containing all pairs of countries
    which appear in the headline along with comma seperated context words
    (all of the words that are NOT the identified country entites)

    Example, if the text is:
        [ctxt_1] [country_1] [ctxt_2] [country_2] [ctxt_3] [country_3] [ctxt_4]

    Then the following 3C2=3 tuples will be created:
        country_1, country_2, with ctxt_[1,2,3,4]
        country_1, country_3, with ctxt_[1,2,3,4]
        country_2, country_3, with ctxt_[1,2,3,4]

    INPUT:
        idx:        integer for keeping track of preprocessing progress
        line:       tuple containing (date string, text of headline) 
        queue:      object to handle writing to output file
        translator:     dictionary of ascii to substituion ascii mappings
        countryforms:   dict mapping textforms to proper country code   
    OUPUT:
        None directly, but writes tuples to textfile based on queue
    """
    # Seperate the tuple
    date, headline = line

    # Lowercase, remove based on trans, trim sides, and split by space for headline
    split_headline = headline.lower().translate(translator).strip().split(" ")

    # Extract all the entities and keep a list of complete context
    entities = []
    context = ""
    for i, token in enumerate(split_headline):
        if token in countryforms:
            entities.append(countryforms[token])
        else:
            context += token + ","
    context = context.strip(",")

    # If at least two entities, append all comb with whole context
    if len(entities) > 1:
        for c1, c2 in combinations(entities, 2):
            # Ensure alphabetical c1, c2 ordering
            if c1 > c2:
                c1, c2 = c2, c1

            # Add the tuple to be written on the queue
            queue.put((idx,"{}\t{}\t{}\t{}\n".format(
                date, c1, c2, context)))
