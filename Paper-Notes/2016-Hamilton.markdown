# Paper info
2016 - William L. Hamilton, Jure Leskovec, Dan Jurafsky
Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change
Accessed as paper for embeddings over time

# Summary

# Misc. keywords
- Diachronic: how things change over time
- Polysemy: existance of multiple meanings

# Skim notes
- proposing two statistical laws related to semantic change:
    - law of conformality: rates of semantic change scale with a negative power
    - law of innovation: after controlling for frequency, polysemous words have 
	significantly higher rates of semantic change
- Semantic change measured as pair-wise and self cosine similarity over time
- SGNS and SVD can arbitraily perform orthogonal transformations so need to align
    between time steps (maybe not an issue with joint optimization)
- validitated shifts by referencing real shifts seen in literature
