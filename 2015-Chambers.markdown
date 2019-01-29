# Paper Info
2015 - Chambers\
Identifying Political Sentiment between Nation States with Social Media\
Accessed as paper referencing 2013 O'Connor

# Summary
Used a combination of filters and linguistic features to find mentions of country
actors in tweets and performed a neg,neut,pos labelled sentiment analysis on the data.
Results were decent (60-80% accuracies).

# Misc. keywords
- antecedent: preceding in time/logical order
- anaphor: a word or phrase that refers to an earlier word or phrase

# Skim notes
- *Reference Resolution* filter out irrelevant discussion 
    - Dining Classifier (if country name used in relation to describing food
        vs. the actual country itself) w/ Logistic Regression
    - Irrelevancy Classifier (catching non-nation geolocation and proper nouns
        which include nation names) w/ Logistic Regression
    - Bootstrap Learner for filtering out sports discussions w/ pointwise
        mutual information scores
- *Contextual Sentiment Analysis* using various features on labelled data
    - labels are $\{neg, pos, objective\}$
    - Model is MaxEnt classifier with default settings using Stanford CoreNLP
- *Full Pipe* 
    1. id country of origin using GPS or profile location
    2. filter irrelevant tweets
    3. classify with contexual sentiment analysis
- Experiments:
    - compared pos/neg ratio gathered from model to opinion polls, military 
        data, and current current formal alliances
    - 0.80 correlation with polls
    - 68-81% accuracy for predicting MIDS negativity
    - 84% accuracy for predicting positive relations based on formal

