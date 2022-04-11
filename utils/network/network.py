import numpy as np
import os
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
import pandas as pd

import networkx as nx

    
def load_correlation_matrix(fname, corr_thresh=0.3):

	# correlation matrix is stack of lists
	#  each lists has cell_id1, cell_id2, correlation value, pvalue
    cc = np.load(fname)

	#
    n_neurons = int(max(int(np.max(cc[:,0])),int(np.max(cc[:,1]))))+1
    cm = np.zeros((n_neurons, n_neurons))

    #
    for k in range(cc.shape[0]):# , desc='unpacking correlation lists..'):
        if cc[k,3]<1E-1:
            cm[int(cc[k,0]), int(cc[k,1])] = cc[k,2]

	# zero out all values < min corr threshold
    #corr_thresh = 0.3
    idx = np.where(cm>=corr_thresh)
    cm  = cm *0
    cm[idx] = 1

	# NOTE Latest correlation matrix loops is not upper triangle only;
	#   loops through all the pairwise vars
    # symmetrize matrix by adding Transpose.
    #cm = cm+cm.T
    
    return cm
    
#
def generate_graph_from_connected_nodes(cm):

    adjacency_matrix = cm.copy()
    
    
    if True:
        G = nx.from_numpy_matrix(adjacency_matrix, create_using=nx.MultiGraph)
    
    else:
        rows, cols = np.where(adjacency_matrix == 1)
		
		#
        edges = zip(rows.tolist(), cols.tolist())
        G = nx.Graph()
        G.add_edges_from(edges)
    
    
    #nx.draw(G, node_size=50, 
    #        #labels=mylabels, 
    #        #with_labels=True
    #       )
    #plt.show()


    return G    
    
#
def get_degree_distribution(G):
    degrees = []
    for d in G.degree:
        degrees.append(d)

    # 
    degrees  = np.vstack(degrees)
    idx = np.argsort(degrees[:,0])
    degrees = degrees[idx]

    ds = degrees[:,1]
    y = np.histogram(ds, bins=np.arange(0,200,2))

    
    return y[1][:-1],y[0]


def get_animal_sessions(animal_id):
    df = pd.read_excel('/media/cat/4TB/donato/CA3 Wheel Animals Database.xlsx')

    #
    binarization_method='upphase'

    idx = np.where(df['Mouse_id']==animal_id)[0].squeeze()
    sessions = df.iloc[idx]['Session_ids'].split(',')
    #print ("sessions: ", sessions)
    #
    cmap = plt.get_cmap("viridis", 10)
    ctr=0
    for k in range(len(sessions)):
        sessions[k] = sessions[k].replace(' ','')


    return sessions

    
