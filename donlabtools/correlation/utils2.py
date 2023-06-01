
import matplotlib

#
import matplotlib.pyplot as plt

# 
import numpy as np
import os
import scipy

# add root directory to be able to import packages
# todo: make all packages installable so they can be called/imported by environment
import sys
module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import networkx as nx


from utils.calcium import calcium


class VisualizeCorrelations():

    #
    def __init__(self, root_dir, animal_id, session):
        
        #
        self.root_dir = root_dir
        self.animal_id = animal_id
        self.session = session
        
        #
        self.main_dir = os.path.join(self.root_dir,
                                self.animal_id,
                                self.session,
                                )
    # 
    def load_corrs(self):

        #
        if self.shuffle==False:
            data_dir = os.path.join(self.root_dir, 
                            self.animal_id, 
                            self.session, '002P-F', 'tif', 'suite2p', 'plane0', 
                            'correlations'
                            )
        else:
            data_dir = os.path.join(self.root_dir, 
                            self.animal_id, 
                            self.session, '002P-F', 'tif', 'suite2p', 'plane0', 
                            'correlations_shuffled'
                            )
    
    
        text = 'all_states'
        if self.subselect_moving_only:
            text = 'moving'
        elif self.subselect_quiescent_only:
            text = 'quiescent'

        data_dir = os.path.join(data_dir, text)

        #
        if self.zscore:
            data_dir = os.path.join(data_dir, 'zscore')
        else:
            data_dir = os.path.join(data_dir, 'threshold')


        # load good cell ids
        fname_ids = os.path.join(data_dir,
                                'good_ids_post_deduplication_upphase.npy')
        self.ids = np.load(fname_ids, allow_pickle=True)

        # load correlation matrix from each file
        fnames_cells = os.listdir(os.path.join(data_dir,
                                                'correlations'))
        fnames_cells.sort()

        corrs = np.zeros((len(fnames_cells),len(fnames_cells))) + np.nan
        pvals = np.zeros((len(fnames_cells),len(fnames_cells))) + np.nan
        corrs_z = np.zeros((len(fnames_cells),len(fnames_cells))) + np.nan

        for fname in fnames_cells:

            # 
            d=np.load(os.path.join(data_dir, 'correlations',fname), allow_pickle=True)
            id = d['id']            
            corr = d['pearson_corr']
            pval = d['pvalue_pearson_corr']
            corr_z = d['z_score_pearson_corr']

            #
            corrs[id] = corr
            pvals[id] = pval
            corrs_z[id] = corr_z

        # delete np.nans from the 2D array
        if self.zscore:
            # so we only use zscore to find edges
            corrs = corrs_z.copy()
            idx = np.where(corrs_z<self.zscore_threshold)
            corrs[idx]=np.nan
        else:
            idx = np.where(pvals>self.pval_threshold)
            corrs[idx]=np.nan
            idx = np.where(corrs<self.pearson_threshold)
            corrs[idx]=np.nan

        # set corrs diagonal to nan
        np.fill_diagonal(corrs, np.nan)

        # count # of non-nan values in 2d array corrs
        print ("# of non-nan values: ", np.count_nonzero(~np.isnan(corrs)), ", as percent of total: ", np.count_nonzero(~np.isnan(corrs))/corrs.size)

        # count # of non-nan values in 2d array corrs that are > 0.3
        #print ("# of non-nan values > 0.3: ", np.count_nonzero(~np.isnan(corrs[corrs>0.3])))


        self.corrs = corrs

        if False:
            #
            plt.figure()
            plt.imshow(corrs)
            cbar = plt.colorbar()
            cbar.set_label('Correlation', rotation=90)

            plt.suptitle(os.path.split(fname_corrs)[0])
            plt.xlabel("Neuron ID")
            plt.ylabel("Neuron ID")
            plt.show()

    def show_histogram(self):

        plt.figure()
        corrs_f = self.corrs.flatten()
        min = np.nanmin(corrs_f)
        max = np.nanmax(corrs_f)
        plt.hist(self.corrs.flatten(), bins=np.arange(min, max,0.1),width=0.09)
        #plt.suptitle(os.path.split(self.fname_corrs)[0])
        plt.xlabel("pairwise correlation / zscore")
        plt.semilogy()

        plt.show()


    def plot_graph_degree(self):
    
        #
        dcorr = self.corrs.copy()

        #
        idx1 = np.where(np.isnan(dcorr))
        idx2 = np.where(np.isnan(dcorr)==False)
        dcorr[idx2]=1
        dcorr[idx1]=0

        #
        sums = np.nansum(dcorr,axis=0)

        # order sums by value
        idx = np.argsort(sums)[::-1]
        sums = sums[idx]
        #print (sums)

        plt.figure(figsize=(8,6))
        ax=plt.subplot(111)
        # increas the fontsize of the ticks
        ax.tick_params(axis='both', which='major', labelsize=20)
        #plt.rcParams['figure.figsize'] = [10, 5]
        plt.scatter(np.arange(len(sums)), sums)
        #plt.suptitle(os.path.split(fname)[0])
        plt.xlabel("Neuron ID")
        plt.ylabel("Network size (>0.3 Pcorr)")
        plt.show()

    def plot_network(self):



        # delete np.nans from the 2D array
        #idx_del = np.where(np.isnan(corrs[0]))[0]
        #corrs = np.delete(corrs, idx_del, axis=0)
        #corrs = np.delete(corrs, idx_del, axis=1)

        # if self.subsample>1:
        #         idx = np.random.choice(np.arange(self.corrs.shape[0]), size=self.subsample, replace=False)
        #         corrs_local = self.corrs[idx].copy()
        #         corrs_local = self.corrs[:,idx].copy()
        # else:
        corrs_local = self.corrs.copy()

        # make graph by adding pairs of connected nodes from corrs array
        G = nx.Graph()
        for k in range(corrs_local.shape[0]):
                for p in range(k+1,corrs_local.shape[1],1):
                        if np.isnan(corrs_local[k,p])==False:
                                G.add_edge(k,p)

        # print number of nodes and edges
        n_nodes = G.number_of_nodes()
        print ("Number of nodes:", G.number_of_nodes())
        print ("Number of edges:", G.number_of_edges())
        
        # remove self-loops
        G.remove_edges_from(nx.selfloop_edges(G))
        
        # make random array up to subsample from size of self.ids
        if self.subsample<n_nodes:
                idx = np.random.choice(G.nodes(), 
                               size=n_nodes-self.subsample, 
                               replace=False)
                print ("removing ", idx.shape[0], " nodes from graph of original size ", n_nodes)
                print ("Number of nodes:", G.number_of_nodes())
                G.remove_nodes_from(idx)
                print ("Number of nodes:", G.number_of_nodes())

        # print number of nodes and edges
        print ("Number of nodes:", G.number_of_nodes())
        print ("Number of edges:", G.number_of_edges())

        # print number of single isolates
        print ("Number of single isolates:", nx.number_of_isolates(G))

        # remove the isolates from the graph
        if self.remove_isolates:
                G.remove_nodes_from(list(nx.isolates(G)))

        # compute density manually
        density = 2*G.number_of_edges() / (G.number_of_nodes() *(G.number_of_nodes() - 1))
        print ("Graph density (manual):", density)

        # 
        plt.figure(figsize=(10,10))

        from pylab import rcParams
        rcParams['figure.figsize'] = 14, 10
        pos = nx.spring_layout(G, scale=20, k=3/np.sqrt(G.order()))
        d = dict(G.degree)
        nx.draw(G, pos, 
                node_color='lightblue', 
                with_labels=False, 
                node_size = 10,
                width = 0.15,
                nodelist=d, 
                alpha=1,
                #node_size=[d[k]*300 for k in d]
                )

        # title should show graph density and also the number of nodes
        plt.suptitle(self.main_dir+'\n'+'Graph density: '+str(np.round(density,3))+'\n'
                                        +'Number of nodes: '+str(G.number_of_nodes())+'\n'
                                        +'Number of edges: '+str(G.number_of_edges())+'\n'
                                        +'Number of isolates: '+str(nx.number_of_isolates(G)),
                                        fontsize=14)


        plt.show()