# Genomics analysis 
The custom code used to analyze genomics data for the paper "Putative Looping Factor ZNF143/ZFP143 is an Essential Transcriptional Regulator with No Looping Function"

# Setup
To run the python scripts provided, first set up a `conda` environment with the packages specified in `environment.yml`. This can be done with the command

```
conda env create -f environment.yml
```

The environment can then be activated using 

```
conda activate genomics_analysis
```

To make the environment available to jupyter notebooks, run the following commands 
```
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=genomics_analysis
```
and use the `genomics_analysis` kernel.

Alternatively, the environment can be setup by installing the listed packages from the necessary source by hand in a fresh conda environment. To run the jupyter notebooks, use the `jupyter notebook` command in a conda environment with jupyter installed. 

# Included files
The following lists give the included scripts with their functions:

#### Python scripts
- `hicrep.py` : This script contains functions used to perform reproducibility analysis for Micro-C. It is used in the hicrep jupyter notebook.
- `pileup_analysis.py` : This script contains functions used to perform pileup analysis/APA for Micro-C. It is used in the pileup_analysis jupyter notebook.
- `region_visualization.py` : This script contains functions used to perform region visualization for Micro-C, ChIP-seq, and PRO-seq. It is used in the region_visualization jupyter notebook.

#### Jupyter notebooks
- `hicrep` : This jupyter notebook was used to generate the SCC correlation matrices for the Micro-C.
- `insulation_score_analyis` : This jupyter notebook was used to generate insulation score tracks and visualize regions for insulation scores.
- `loop_strength_correlations` : This notebook was used to calculate loop strengths on an individual loop basis and make correlation plots. 
- `p_of_s_and_compartment_calls` : This notebook was used for P(s) curve plotting and compartment calls.
- `pileup_analysis` : This notebook was used for the pileup analysis.
- `region_visualization` : This notebook was used to visualize regions with Micro-C, ChIP-seq, and PRO-seq data. 