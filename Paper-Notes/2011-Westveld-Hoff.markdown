# Paper info
2011 - Anton H. Westveld, Peter D. Hoff
A Mixed Effects Model for Longitudinal Relational and Network Data with Applications
to Interantional Trade and Conflict\
Accessed as paper discussed during meeting

# Summary
Used linear models to model dyadic trade/conflict levels between countries using
GDP, geographical distance, democracy scores, cooperation in conflict, and 
shared governmental organizations as variables.

# Misc. keywords
- Polity: a measurement of a countries "democracy" level 0 to 20 from a popular
    polity data set used commonly in political science
- CC: Cooperation in conflict measures active military cooperation

# Skim notes
- International Trade Model:
    - Following a gravity model, a trade linear model taking into account GDP, distance,
    polity, CC is considered (similair to traditional gravity model in polisci),
    note all of these vary over time other than distance
    - Data is international trade between 58 countries during 1981=2000 (can be 
        found in (2010 Westveld and Hoff)) can also be seen in section 5
	- Data header is "","i.j.t.exp.imp.ltrade.lgdp.exp.lgdp.imp.ldist.pty.exp.pty.imp.cc" 
	    - first col is UUID of row
	    - i is exporter ID
	    - j is importer ID
	    - t is time (1-20, repping 1981-2000)
	    - exp is exporter name
	    - imp is importer name
	    - ltrade is ln(trade) from i to j (presumably units are ln($))
	    - lgdp.exp and lgdp.imp are the ln(GDP)'s of the export and importer respectivly
	    - ldist is the ln(distance) between i and j
	    - pty.exp and pty.imp are the polity scores of i and j respectivly
	    - cc is ??? doesn't seem useful
    - Out of sample prediction procedure:
	- Randomly split to train-test 75-25
	- calc MSE on hidden trained on 75
    - Models learned through MCMC
- Military Conflict Model:
    - Similair to International Trade model but also with level of alliance as a variable
        in the model (along with exports, imports, # inter-govermental org. between
        the dyad, and dist)
