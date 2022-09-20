import numpy as np
import os
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from scipy import signal

#
def fix_jumps(my_data,
              min_DLC_likelihood,
              max_distance_jump,
              min_length_seg):


    #
    #print (my_data[2])
    xs = my_data[:,::3]
    ys = my_data[:,1::3]
    likelihoods = my_data[:,2::3]

    #
    xs_median = np.nanmedian(xs,axis=1)
    ys_median = np.nanmedian(ys,axis=1)
    locs = np.vstack((xs_median,
                      ys_median)).T

    #
    
    #
    print ("locs: ", locs.shape)
    #print (locs)

    #
    diff = locs[1:]-locs[:-1]
    print ("diff: ", diff.shape)
    
    #
    dist = np.linalg.norm(diff,axis=1)
    print ("dist: ", dist.shape)

    ###############################################
    ############## DELETE DLC ERRORS ##############
    ###############################################
   
    # delete low prob detections
    threshold = min_DLC_likelihood
    idx = np.where(likelihoods<threshold)
        
    xs[idx]=np.nan
    ys[idx]=np.nan
    xs_median = np.nanmedian(xs,axis=1)
    ys_median = np.nanmedian(ys,axis=1)

    ###############################################
    ########### DISCONNECT BIG JUMPS ##############
    ###############################################
    threshold = max_distance_jump
    locs = np.vstack((xs_median,
                      ys_median)).T

    #
    diff = locs[1:] - locs[:-1]
    dist_fixed = np.linalg.norm(diff, axis=1)

    #
    idx = np.where(dist_fixed > threshold)[0]
    for id_ in idx:
        xs_median[id_:id_+1] = np.nan
        ys_median[id_:id_+1] = np.nan

    ###############################################
    ########### DELETE SHORT SEGMENTS #############
    ###############################################
    #min_length_seg = 5

    # detect bouts
    def using_clump(a):
        return [a[s] for s in np.ma.clump_unmasked(np.ma.masked_invalid(a))]

    # find nan locations
    idx = np.where(np.isnan(xs_median))[0]
    temp = np.arange(xs_median.shape[0],dtype=np.float32)
    temp[idx] = np.nan

    # return all non-nan bouts
    bouts = using_clump(temp)

    #
    for bout in bouts:
        if bout.shape[0]<min_length_seg:
            #print("deleting bout of length: ", xs_median[np.int32(bout)].shape)
            xs_median[np.int32(bout)] = np.nan
            ys_median[np.int32(bout)] = np.nan


    ###############################################
    ############# FILL IN ALL NANS ################
    ###############################################

    idx = np.where(np.isnan(xs_median))[0]
    print("# of nans: ", idx.shape)

    n_steps = 5
    for id_ in idx:
        xs_median[id_] = np.nanmedian(xs_median[id_-n_steps:id_+n_steps])
        ys_median[id_] = np.nanmedian(ys_median[id_-n_steps:id_+n_steps])

    idx = np.where(np.isnan(xs_median))[0]
    print ("# of nans: ", idx.shape)

    #####################################
    ######## RECONSTITUTE TRACK #########
    #####################################
    #
    locs = np.vstack((xs_median,
                      ys_median)).T
    
    print ("locs fixed: ", locs.shape)

    #
    #return dist, dist_fixed, locs
    return locs
