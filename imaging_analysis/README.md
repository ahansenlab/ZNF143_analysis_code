# Imaging analysis 
The custom code used to analyze imaging data for the paper "Putative Looping Factor ZNF143/ZFP143 is an Essential Transcriptional Regulator with No Looping Function"

# Setup
To run the python scripts provided, first set up a `conda` environment with the packages specified in `environment.yml`. This can be done with the command

```
conda env create -f environment.yml
```

The environment can then be activated using 

```
conda activate imaging_analysis
```

To make the environment available to jupyter notebooks, run the following commands 
```
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=imaging_analysis
```
and use the `imaging_analysis` kernel.

Alternatively, the environment can be setup by installing the listed packages from the necessary source by hand in a fresh conda environment. To run the jupyter notebooks, use the `jupyter notebook` command in a conda environment with jupyter installed. 

# Included files
The following lists give the included scripts with their functions:

#### Python scripts
- `fast_spt_analysis.py` : This script contains functions used to perform the fast spt analysis. It is used in the fast_spt_analysis jupyter notebook. 
- `frap_analysis.py` : This script contains functions used to perform the FRAP analysis. It is used in the frap_analysis jupyter notebook.

#### Jupyter notebooks
- `fast_spt_analysis` : This jupyter notebook was used to analyze the fast spt data. 
- `frap_analysis` : This jupyter notebook was used to analyze the FRAP data. 
