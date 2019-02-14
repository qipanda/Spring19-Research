# Paper info
2017 - Rebecca S. Portnoff et al.\
Tools for Automated Analysis of Cybercriminal Markets\
Accessed as paper referencing 2013 O'Connor

# Summary
Developed methods and tools to monitor underground cybercriminal markets on
online forums at an aggregate level using NLP techniques. 

# Misc. keywords

# Skim notes
- Want to extract three properties from each forum post:
    1. type of transaction (buy, sell, curr_exg)
    2. product
    3. price/exg_rate
- Mainly used an SVM model with text-based features like n-grams and char-grams
- Able to get high level characteristics of each forum based on top 10 products
    detected by the models
- Case studies demonstrated use of these models for finding the popularity of 
    original vs. bulk/hacked accounts and what currencies are in high demand
