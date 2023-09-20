# manifolds project @ Donato lab

## Installation

### Recommended steps 
#### Create Conda Environment
The code snippets corresponding to the numbers are found below. Copy all lines at once or line by line

0. Open a command-line interface preferably inside the conda enviroment on your operating system (Windows, Mac)
1. [Install conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html)
2. Use command-line to make a conda enviorment for running manifolds project
3. Activate environment
4. Install dependencies
5.  Install jupyter notebook 

```bash
conda create -n manifolds python=3.8;
conda activate manifolds;
pip install matplotlib numpy scipy tqdm scikit-learn parmap networkx pandas opencv-python;
pip install jupyter
```

#### Run Jupyter notebookym
1. Download and unzip the [Donatolab binarization repo](https://github.com/donatolab/manifolds)

![GitHub Green Button](https://camo.githubusercontent.com/2ee2e59ced868d6a7653f3086ff39507c5330999cf127eae8a71e741bfb78a1a/68747470733a2f2f692e6962622e636f2f336d4c6e4b4d482f636c6f6e652e706e67 "Click on such a button in the link above")

![GitHub Download Zip](https://camo.githubusercontent.com/3aa5742481d5d286ecfa18e2f716e008d144f72fb4988919d65db05e83636ae0/68747470733a2f2f692e6962622e636f2f334d3543584b6d2f636c6f6e652d7a69702e706e67)

2. Start jupyer notebook by typing it in at the command line

```bash
jupyter lab
```

3. Navigate to the folder where the code unzipped and click on this file to start the jupyter notebook:

```bash
"donlabtools/binarization/Binarize_Inscopix_V3.ipynb" 
```
4.  Run the first cell and then input the location of your suite2p folder in 2nd cell. Run the rest of the notebook.

5.  The code will save 2 files: 
    1.  binarized_traces.npz (a python numpy file) and 
    2.  binarized_traces.mat (a matlab file).
    3.  Additional plots