# E.-coli-Min-system-1D-simulation
[![DOI](https://zenodo.org/badge/660496460.svg)](https://zenodo.org/badge/latestdoi/660496460)

Randomly produced 4 rate constants to have all positive particle number and pass linear stability analysis and have Min oscillation in one dimensional lattice. Additionally, adding folders for producing Figure 3 in the paper.

# Description

Four python 3.9 scripts are available in this repository. Scripts producing Figure 3 are *.m. 

# Instructions

1. run c1.Min.LSA.random.para.py to get one parameter set.
2. with the parameter set produced in the last step, get the time and position data by running c2.get.timecourse.py. Simulation time is 140 seconds.
3. c3.kymograph.py produces a figure with membrane MinD as the target, different cell lengths are 1.6, 1.8, ..., 2.3 micrometer. Time in the plot is from 40 to 140 seconds.
4. c4.analysis.py produce a figure with three panels as the left panel is periods vs cell length, the middle panel as lambda value vs cell length and bi-phasic linear fitting. The right panel is showing the Iratio vs cell length.

# References

1.  Fange D, Elf J. Noise-Induced Min Phenotypes in E. coli. PLOS Comput Biol. 2006;2(6):e80. doi:10.1371/journal.pcbi.0020080

2.	Halatek J, Frey E. Highly Canalized MinD Transfer and MinE Sequestration Explain the Origin of Robust MinCDE-Protein Dynamics. Cell Rep. 2012;1(6):741-752. doi:10.1016/j.celrep.2012.04.005

3.	Wu F, van Schie BGC, Keymer JE, Dekker C. Symmetry and scale orient Min protein patterns in shaped bacterial sculptures. Nat Nanotechnol. 2015;10(8):719-726. doi:10.1038/nnano.2015.126

4.	Cross M. Notes on the Turing Instability and Chemical Instabilities. Published online 2006.


# Feedback

Made changes to the layout templates or some other part of the code? Fork this repository, make your changes, and send a pull request.
Do these codes help on your research? Please cite as the follows: Growth-dependent concentration gradient of the ....C Parada, CCS Yan, CY Hung, IP Tu, CP Hsu, YL Shih.
https://doi.org/10.1101/2023.08.01.551406
