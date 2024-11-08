[![DOI](https://zenodo.org/badge/768283056.svg)](https://doi.org/10.5281/zenodo.14056922)

# Putative Looping Factor ZNF143/ZFP143 is an Essential Transcriptional Regulator with No Looping Function

This github repository contains the code used for analysis in the paper "Putative Looping Factor ZNF143/ZFP143 is an Essential Transcriptional Regulator with No Looping Function". Code is provided either in the form of Python scripts or as Jupyter notebooks to be run in conda environments containing the required packages.

## Summary of contents
#### frap_drift_correction
This folder provides the code used to perform the drift correction for the FRAP movies. It is meant to be installed as a package and then used from the command line.

#### genomics_analysis
This folder provides the code used to analyze the genomics data. A description of the scripts it contains is as follows: 
- `hicrep.py` : This script contains functions used to perform reproducibility analysis for Micro-C. It is used in the hicrep jupyter notebook.
- `pileup_analysis.py` : This script contains functions used to perform pileup analysis/APA for Micro-C. It is used in the pileup_analysis jupyter notebook.
- `region_visualization.py` : This script contains functions used to perform region visualization for Micro-C, ChIP-seq, and PRO-seq. It is used in the region_visualization jupyter notebook.

- `hicrep` : This jupyter notebook was used to generate the SCC correlation matrices for the Micro-C.
- `insulation_score_analyis` : This jupyter notebook was used to generate insulation score tracks and visualize regions for insulation scores.
- `loop_strength_correlations` : This notebook was used to calculate loop strengths on an individual loop basis and make correlation plots. 
- `p_of_s_and_compartment_calls` : This notebook was used for P(s) curve plotting and compartment calls.
- `pileup_analysis` : This notebook was used for the pileup analysis.
- `region_visualization` : This notebook was used to visualize regions with Micro-C, ChIP-seq, and PRO-seq data.

#### imaging_analysis
This folder provides the code used to analyze the imaging data apart from performing the FRAP drift correction. A description of the scripts it contains is as follows:
- `fast_spt_analysis.py` : This script contains functions used to perform the fast spt analysis. It is used in the fast_spt_analysis jupyter notebook. 
- `frap_analysis.py` : This script contains functions used to perform the FRAP analysis. It is used in the frap_analysis jupyter notebook.

- `fast_spt_analysis` : This jupyter notebook was used to analyze the fast spt data. 
- `frap_analysis` : This jupyter notebook was used to analyze the FRAP data.

## How to cite

This work is shared under an MIT license. If you make use of analysis scripts or data from this work, please cite as follows:

Narducci, D.N. & Hansen, A.S. Putative Looping Factor ZNF143/ZFP143 is an Essential Transcriptional Regulator with No Looping Function. 
