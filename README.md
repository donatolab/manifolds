# manifolds project @ Donato lab

### Installation

Recommended steps


0.  Open a command-line interface preferably inside the conda enviroment on your operating system (Windows, Mac)

1.  Use command-line to make a conda enviorment for running manifolds project:

conda create -n manifolds python=3.8

2.  Activate environment:

conda activate manifolds

3. Install dependencies (might have to do them 1 at a time; eventually will have a script for this)

pip install: matplotlib, os, numpy, scipy, tqdm, sklearn, pickle, parmap, networkx, pandas, cv2

4.  Install jupyter notebook 

conda install -c anaconda jupyter

5. Download and unzip the Donatolab binarization repo (https://github.com/donatolab/manifolds, click onthe green "Code" button)

6. Start jupyer notebook by typing it in at the command line

jupyter notebook
 
7. Navigate to the folder where the code unzipped and click on this file to start the jupyter notebook:

"Binarize_Suite2p_Inscopix.ipynb" 

8.  Run the first cell and then input the location of your suite2p folder in 2nd cell. Run the rest of the notebook.

9.  The code will save 2 files: binarized_traces.npz (a python numpy file) and binarized_traces.mat (a matlab file).

10(Optional) You can then use the last cell to visualize the traces and binarized versions for any specific cell. 
