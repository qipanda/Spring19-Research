# Paper info
2014 - Shahryar Minhas, Jay Ulfelder, Michael D Ward
Mining texts to efficiently generate global data on political regime types
Accessed as paper referencing 2013 O'Connor

# Summary
Trained SVM classifiers to infer regime type from textual data sources. Was done
for each country on an annual timestep basis.

# Misc. keywords

# Skim notes
- (Hegre, 2014) apparently shows that national regime type shapes conutries propensity
    to go to way with other countries or their own citiznes
- (Goldstone et al, 2010) showed that regime type is the single most powerful
    predictor of onsets of national political crises such as civil wars, coups, 
    and state collapse
- This paper generates binary measures of democracy, military rule, one-party rule
    and monarchy
- Other possibly interesting papers from background:
    - Generating political event data:
        - D'Orazio et al, 2014
        - King and Lowe, 2003
    - Measuring political tension:
        - Cahdefaux, 2014
    - Measuring partisan affiliation:
        - Slapin and Proksch, 2010
        - Yu et al, 2008
    - Measuring legislative agendas:
        - Grimmer, 2010
- Preprocessing removed punctuation, stop words, numbers, proper nouns, acronyms,
    lemmatizing (getting canonical form of word)
- Structured data for learning is a set of docs per year-country:
    - do BOW + tf-idf as feature set for this year-country
