# manifolds project @ Donato lab

### Installation

Recommended steps


0.  Open a command-line interface preferably inside the conda enviroment on your operating system (Windows, Mac)

1.  Use command-line to make a conda enviorment for running manifolds project:

conda create -n manifolds python=3.8

2.  Activate environment:

conda activate manifolds

3. Install depencies:

pip install matplotlib, os, numpy, scipy, tqdm, sklearn, pickle, parmap, networkx, pandas, cv2

4.  Install jupyter notebook 

conda install -c anaconda jupyter

5.  Start jupyer notebook

jupyter notebook

6.  Navigate to the folder and click on this file to start the jupyter notebook:

"Binarize_Suite2p_Inscopix.ipynb" 

7.  Run the first cell and then input the location of your suite2p folder in 2nd cell. Run the rest of the notebook.

8.  The code will save 2 files: binarized_traces.npz (a python numpy file) and binarized_traces.mat (a matlab file).

9.  (Optional) You can then use the last cell to visualize the traces and binarized versions for any specific cell. 
