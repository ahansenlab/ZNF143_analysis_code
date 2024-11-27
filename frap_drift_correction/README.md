# FRAP Analysis
FRAP Analysis is a Python gui which enables analysis of FRAP data. Currently, FRAP Analysis is optimized for circle FRAP. It implements background subtraction, photobleaching correction, and drift correction. Additionally, it implements circle FRAP models from Axelrod et al. 1976, Soumpasis et al. 1983, Sprague et al. 2004, and Mueller et al. 2008. 

## Installation:
1. Clone the repository.

2. Create a `conda` environment for `FRAP_Analysis`. (If you don't already have it, you'll need `conda`: https://docs.conda.io/en/latest/miniconda.html.) Navigate to the top-level `FRAP_Analysis` directory and run 

```
    conda env create -f requirements.yml
```

3. Switch to the `frap_env` environment:

```
    conda activate frap_env
```

4. Finally, install the `FRAP_Analysis` package. From the top-level `FRAP_Analysis` directory, run

```
    python setup.py develop
```
## Running the GUI:
To launch the GUI, first switch to the `FRAP_Analysis` environment:

```
    conda activate frap_env
```

Then start the main GUI with

```
    frap main Samples/Example_Movie.czi
```
