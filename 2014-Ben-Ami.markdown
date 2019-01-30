# Paper info
2014 - Zvi Ben-Ami, Ronene Feldman, Binyamin Rosenfeld
Entities' Sentiment Relevance
Accessed as paper referencing 2013 O'Connor

# Summary
Explores variants of sentiment relevance detection (which entity does the sentiment
in a piece of text refer to) showing experimentally that the best methods of solving
this task utilize document-level information about the target entities.

# Misc. keywords

# Skim notes
- relevance detection in the context of this paper can be thought of as a binary
    choice between 'relevant' & 'irrelevant' given a text document, sentiment
    expression, and target entity (is this sentiment expression relevant to target
    entity)
    - authors utilize entity "status" within the document (Target, Accidental,
        RelationTarget, or ListTarget, these are manually annotated)
- input contains document text, sentiment expressions, and all coreferences of 
    the entity
- Methods compared:
    1. Baseline: Every expression is relevant
    2. Physical-proximity-based: based on literal position of text to entity
    3. Syntactic-proximity-based: based on dependency parse position of text to entity
    4. Classification-based: linear classifier using anything in the input as features
        (Large Margin training regularized perceptron Scheible and Schutze 2013)
    5. Sequence-classification-based: same as above, but considers each sentiment
        expression in the document as a sequence per document (probabilistic sequence
        classifier. CRF Lafferty et al. 2001)
- Experiment results:
    1. Showed that the detected relevance is important for more typical sentiment
        analysis (vs. just using the whole document) in terms of precision, but
        bad recall
    2. Showed that entity status is an important feature by using variations of
        classifiers tailored for each and trying them on the subsets they were
        and were not meant for (no surprise, they performed best on subsets they
        were meant for in terms of entity status)
    3. Showed that entity status can be automatically extracted with minimal loss
        in performance (exact method is unclear?)
    4. Overall sequence classification and classification in general was better at
        relevance detection
