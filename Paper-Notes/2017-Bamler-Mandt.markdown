# Paper info
2017 - Robert Bamler, Stephan Mandt
Dynamic Word Embeddings
Accessed as a reference to 2016-Hamilton (Embeddings over time)

# Misc. keywords

# Skim notes
- Joint training instead of learning embeddings per time slice:
    - Claim leads to better interpretability and higher predictive likelihoods
    - references for time-bin static learning:
	- Mihalcea & Nastase, 2012
	- Sagi et al. 2011
	- Kim et al., 2014
	- Kulkkarni et al., 2015
	- Hamilton et al., 2016
    - Claimed problems with time-bin static:
	1. Since word embedding models are non-convex, training twice on the 
	    same data can lead to different results
	2. Dividing corpus into bins leads to training sets that may be too small
	    and we risk over fitting on a small data bins
	3. Since corpus size is not infinite, noise during learning is hard to 
	    differntiate from actual semantic drift
- Dynamic Skip-Gram Model:
    - Uses Kalman filter as a prior for the time-evolution of the latent 
	embeddings (Welch & Bishop, 1995)
    - Diffusion constant D details (Welch & Bishop, 1995)
    - Transitions are defined by Normal Distributions (what are eq 5,6)?
