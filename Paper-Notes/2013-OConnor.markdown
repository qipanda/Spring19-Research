# Paper info
2013 - O'Connor, Stewart, Smith\
Learning to Extract International Relations from Political Context\
Accessed as core paper

# Summary
Created a probabilistic unsupervised model which extracts major political actors
from news corpora. Performance is assessed on recovering expert-assigned event
class valences and detecting real-world conflict. Ideally reflects relevant
events between the actors and can be grouped into meaningful low-dimensional
event types.

# Misc. key words
- *events data:* political scientist term from 1960's for records of public 
    micro-level interactions between major political actors of the form 
    "someone does something to someone else"
- *TABARI:* open-source rule-based event extraction system for political domain 
- *predicate:* part of sentence containing verb which states something about subject
- *frame* is synonomous with *event type*

# Data and preprocessing 
- Model is learned from a corpus of 6.5mil newswire articles from the English
    Gigaword 4th edition (1994-2008, Parker et al., 2009), it is supplemented
    with a sample of data from the NYT Annotated Corpus (1987-2007, Sandhaus, 2008)
- Stanford CoreNLP used to POS-tag and parse articles producing tuples of form:
    $$
    <s,r,t,w_{predpath}>
    $$
    - $s$ and $r$ denote "source" and "receiver" (political entities) from the set
        of $\mathcal{E}$
    - $t$ is the timestep derived from article publishing date
    - $w$ is textual predicate expressed as a dependency path (e.g. accused ->)
- Entities found by finding country names listed in TABARI (235 entities, 2500 names)
- *Q* pg2, left col, "Whenever a name is found..." -> confused about what this is?
- Verb paths id'd by looking at shortest dependency path two mentions in a sent.
    e.g. talk <-(prep_with), fight <-(prep_alongside), reject <-(dobj),
    allegation <-(poss) 
    - dobj means direct object?
    - poss means possessive?
- Focused on directly reported events to remove unverifiable cases of indirect
    reporting or hypothetical reporting 
- *note* should study up more on predicate pathes (dependency grammar?)
- sports and finance news have been filtered using keyword filters
- topic filtering employed some text-classifiers (l1-regularized logistic reg
    with unigram and bigram features)
- tuples were the receiver and source are the same were removed
- 365,623 event tuples from 235,830 documents -> 421 dyads, 10,457 unique 
    predicate paths, 1,149 discrete timesteps of 7 days (1987-2008)

# Model
- modelling:
    $$
    p(w_{predpath}|s,r,t)
    $$
- two sub models
    - Context submodel: encodes how political context affects the prob. dist. 
        over event types
    - Language submodel: how those events are manifested as textual pred. pathes
    - Cond on context, how does event type dist change | cond on event type, how
        does textual predicate pathes dist change
- $K$ is a hyperparameter denoting number of frames

# Experiments
- Lexical Scale Impurity -> Compared learned predicate pathes to gold-standard
    scale scores by matching inferred predicate pathes to those which existed
    in TABARI
- Conflict Detection -> Validate model by seeing if it can based on news text,
    tell if there is armed conflict currently happening, gold standard is MID
- Both measures showed improvment of smoothed model as K increased and did better
    than their baselines
- Qualitative case study -> looking at Israel/Palestine, highest dyad occurance
    pair. Shows that some learned frames have spikes of activity in $E[\theta]$
    corresponding with major political events.
