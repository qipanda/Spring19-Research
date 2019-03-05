# Jan 30th 2019
- Same idea as 2013 but try to infer hierarchical structure of entities?
    -e.g. Country -> Government Agencies -> Individuals
- Public perception of entities (not nec. countries, could be orgs or individuals)
    in some "characteristic" space? Maybe use social media data as well

# Feb 5th 2019
- Compare country-level data (e.g. dem score) with linguistic characteristics
    of news articles mentioning it:
- Searching for overlapping country-data + news article data (uni and dyadic):
    - News Article Data:
        - American News scraped between 2016 and July 2017
            https://www.kaggle.com/snapcrack/all-the-news
        - Million news headlines 
            https://www.kaggle.com/therohk/million-headlines/home,
        - News Headlines of India
            https://www.kaggle.com/therohk/india-headlines-news-dataset
        - NYT API article metadata + headlines
            https://developer.nytimes.com/apis 
        - all links that reference a wikipedia article
            https://code.google.com/archive/p/wiki-links/downloads,
        - pushshift.io submission headers as "article titles" on /r/worldnews
        - all Hacker News comments
            https://www.kaggle.com/hacker-news/hacker-news
        - IrishTimes News headers with category
            https://www.kaggle.com/therohk/ireland-historical-news
        - Newsroom: 1.3 million articles and summaries written by authors/editors
            of newsrooms of 38 major news publication 1998-2017
            https://yoavartzi.com/pub/gna-naacl.2018.pdf
    - Political "score" Data:
        - Polity IV: Coup D'Etat dataset (all coups from 1946-2017 from major countries
            including exact date and success score) 

# Feb 6th 2019
- With header data + coup data, are news articles an indicator of incoming coup in
    the respective country? If so, is there a linguistic shift in the way the country
    is mentioned in the news vs. normally? Does the frequency increase?
- How do other countries interact with the couping country leading up the coup? Is 
    there a notable trend traceable linguistically from news articles/news headers?

# Feb 8th 2019
- Look into word-vector style representation of countries from headlines as a prior
    for a language model describing them in the news?
- Restrict language model to "interacting" words and have priors be for country pairs?

# Feb 9th 2019
- Wrote down current idea of model in notes
    - 1-hot of pairs of countries into K dimension vector of weights representing
        the contry pairs context in latent space
    - Use country pairs K dim vector as input into hidden layers (feed forward
        for now) then feed into a softmax containing vocab and evenly split proportions
        of words in the context

# Feb 27th 2019
- How to make results more interpretable?
- Reduce vocab down to "classes"?
- Define classes based on international relations literature, find closest X words
    to defined class words based on word2vec or some other synonom thing, map
    all words down to that class? (Add structure)
- Motivation of work could be to allow people to transfer their knowledge of one
    relationship to another at a glance

# March 4th 2019
- Retrain model on better preprocessing & make neg sampling more tunable
- Iterate with just one hold out and val set, remember to hold out a test set
- Hyperparameters beyond embedding dim, also neg sampling and subsampling
- Use existing word2vec vectors as initialization?
- Sports filter?
