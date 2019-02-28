import HeadlineProcessor as hp
import string
import pickle
import multiprocessing as mp

def listener(writefilepath: str, printprog: bool, queue: mp.Queue, header: str) -> None:
    """
    Listens for clean tuples to be written to [writefilepath]
    """
    with open(writefilepath, "w") as f:
        f.write(header)
        while True:
            idx, msg = queue.get()
            if printprog:
                print(idx)
            if msg == "KILL":
                break
            f.write(msg)
            f.flush()

if __name__ == "__main__":
    # Get abc data, translator, and countryforms
    with open("../Data/Times-of-India/india-news-headlines.csv", "r") as raw_data:
        lines = [tuple(line.split(",")) for line in raw_data.read().splitlines()][1:]
    with open("../Data/TABARI/countryforms.pickle", "rb") as handle:
        countryforms = pickle.load(handle)
    translator = str.maketrans("", "", string.punctuation) 

    # Get rid of category, make it date, text
    lines = [(line[0], line[2]) for line in lines]

    # Set up multiprocessing
    manager = mp.Manager()
    queue = manager.Queue()
    pool = mp.Pool(processes=10)

    # Start the listener
    writefilepath = "../Data/ABC-News/india-complete-ctxt.txt"
    header = "date\tc1\tc2\tctxt\n"
    printprog = True
    watcher = pool.apply_async(listener, (writefilepath, printprog, queue, header)) 

    # Process headlines and pass to queue for writing 
    results = []
    for idx, line in enumerate(lines):
        result = pool.apply_async(
            hp.writeCompleteTuple, (idx, line, queue, translator, countryforms))
        results.append(result)

    # All async calls must be complete before we kill the queue and pool
    for result in results:
        result.wait()
    
    # Kill listener once done processing headlines and kill the pool
    queue.put("KILL")
    pool.close()
