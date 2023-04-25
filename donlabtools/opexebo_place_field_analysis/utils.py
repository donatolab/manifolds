from scipy.signal import savgol_filter
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import os
import scipy
import matplotlib.pyplot as plt

#
import cv2
import scipy.stats as stats
from scipy.signal import savgol_filter
import glob
#
import parmap

# add root directory to be able to import packages
# todo: make all packages installable so they can be called/imported by environment
import sys
from tqdm import tqdm, trange
import parmap

#
import opexebo as op
import astropy

#########################################################################
#########################################################################
#########################################################################
def get_spikes(locs,
               locs_times,
               spikes_all,
               cell_id):
    #
    idx = np.where(spikes_all[:, 1] == cell_id)[0]

    #
    spikes = spikes_all[idx, 0]
    idx = np.argsort(spikes)
    spikes = spikes[idx]

    #
    print("cell: ", cell_id, ", spike times: ", spikes.shape)

    # make the spikes_tracking vstack;  not completely clear this is correct; X->Y inverstion etc? not clear
    min_time = 0.0333
    min_spikes = 1

    #
    spikes_tracking = []
    ctr = 0
    for k in trange(locs_times.shape[0]):
        temp = locs_times[k]
        idx1 = np.where(np.logical_and(spikes > (temp - min_time), spikes <= temp))[0]

        #
        if idx1.shape[0] >= min_spikes:
            ctr += 1
            spikes_tracking.append([k,
                                    locs[k, 0],
                                    locs[k, 1]])

        # delete spikes before
        if idx1.shape[0] > 0:
            spikes = spikes[idx1[0]:]

    #
    spikes_tracking = np.vstack(spikes_tracking)

    return spikes_tracking.T


#
def get_field(occ_map,
              cell_spikes,
              arena_size):
    #
    rm = op.analysis.rate_map(occ_map,
                                       cell_spikes,
                                       arena_size)

    #
    sigma = 1
    rms = op.general.smooth(rm, sigma)

    #
    init_thresh = 1
    min_bins = 2
    min_mean = rms.max() * 0.1  #
    min_peak = rms.max() * 0.1  #

    #
    fields, fields_map = op.analysis.place_field(rms,
                                                 min_bins=min_bins,
                                                 min_peak=min_peak,
                                                 min_mean=min_mean,
                                                 init_thresh=init_thresh,
                                                 search_method='sep',
                                                 debug=False
                                                 #    limits=limits
                                                 )
    #
    return rms, fields_map


#
def compute_place_field(cell_id,
                        locs,
                        locs_times,
                        spikes,
                        occ_map,
                        arena_size,
                        ):
    #
    cell_spikes = get_spikes(
        # locs_binned,
        locs,
        locs_times,
        spikes,
        cell_id)

    #
    n_spikes = cell_spikes.shape[1]
    print("cell spikes: ", cell_spikes.shape)

    # find mid-point of time:
    mid_pt = locs_times.shape[0] // 2
    mid_pt_idx = np.argmin(np.abs(mid_pt - cell_spikes[0]))
    print("mid pt: ", mid_pt)
    print("mid_pt_idx: ", mid_pt_idx)

    #
    plt.figure(figsize=(10, 10))

    #
    rms, fields_map = get_field(occ_map,
                                cell_spikes,
                                arena_size)

    #
    ax = plt.subplot(3, 3, 1)
    plt.imshow(rms)
    plt.ylabel("rate map (smoothed)")
    plt.title("rms (all)")

    #
    ax = plt.subplot(3, 3, 4)
    plt.imshow(rms / occ_map)
    plt.ylabel("rate map/occ_map")

    #
    ax = plt.subplot(3, 3, 7)
    plt.imshow(fields_map)
    plt.ylabel("fields")

    ########################################
    rms, fields_map = get_field(occ_map,
                                cell_spikes[:, :mid_pt_idx // 2],
                                arena_size)

    #
    ax = plt.subplot(3, 3, 2)
    plt.title("rms (1st half)")
    plt.imshow(rms)

    #
    ax = plt.subplot(3, 3, 5)
    plt.imshow(rms / occ_map)

    #
    ax = plt.subplot(3, 3, 8)
    plt.imshow(fields_map)

    #######################################
    rms, fields_map = get_field(occ_map,
                                cell_spikes[:, mid_pt_idx // 2:],
                                arena_size)
    ax = plt.subplot(3, 3, 3)
    plt.imshow(rms)
    plt.title("rms (2nd half)")

    #
    ax = plt.subplot(3, 3, 6)
    plt.imshow(rms / occ_map)

    ax = plt.subplot(3, 3, 9)
    plt.imshow(fields_map)
    plt.suptitle("Cell: " + str(cell_id) + ", # spks: " + str(n_spikes))

    if True:
        plt.show()
    else:
        plt.savefig('/media/cat/4TB1/temp/place_field_' + str(cell_id) + '.png')
        plt.close()

    return cell_spikes
#
def load_locs_traces(fname,
                     arena_size):
    
    #
    locs = np.load(fname)
    print (locs.shape)

    ####################### COMPUTE SPATIAL OCCUPANCY ###########################
    times = np.arange(locs.shape[0])

    #
    min_x = np.min(locs[:,0])
    max_x = np.max(locs[:,0])

    min_y = np.min(locs[:,1])
    max_y = np.max(locs[:,1])

    #
    locs[:,0] = (locs[:,0]-min_x)/(max_x-min_x)*arena_size[0]
    locs[:,1] = (locs[:,1]-min_y)/(max_y-min_y)*arena_size[1]

    ####################### LOAD SPIKES ########################
    fname = '/media/cat/4TB1/donato/nathalie/DON-007050/FS1/binarized_traces.npz'
    bin_ = np.load(fname,
                   allow_pickle=True)

    #
    upphases = bin_['F_upphase']
    #print ("traces: ", traces.shape, traces[0][:100])
    
    filtered_Fs = bin_['F_filtered']
    
    #
    return locs, upphases, times, filtered_Fs

#
def get_t_exp_from_filtered_F(upphase, 
                              filtered,
                             locs):
    
    ''' function uses upphase detected spiking
        and then uses the value of the filtered [ca]
    '''
    
    # check when upphase spiking occurs
    t = np.where(upphase>0)[0]
    
    print ("locs: ", locs.shape)
    
    tt = []
    x = []
    y = []
    gradient_scaling = 20
    for k in range(t.shape[0]):

        #
        temp_filtered = filtered[t[k]]
        grad = int(temp_filtered)

        # 
        if grad>0:
            tt.append(np.ones(grad))
            x.append(np.ones(grad)*locs[t[k],0])
            y.append(np.ones(grad)*locs[t[k],1])

    tt = np.hstack(tt)
    x = np.hstack(x)
    y = np.hstack(y)
    
    return tt,x,y

#
def get_t_exp_from_gradient(
                            upphase, 
                            filtered,
                            locs
                        ):
    ''' function uses upphase detected spiking
        and then scales the spiking based on the gradient of the filtered [ca]
    '''
    
    t = np.where(upphase>0)[0]
    
    print ("locs: ", locs.shape)
    
    tt = []
    x = []
    y = []
    gradient_scaling = 20
    for k in range(t.shape[0]):

        #
        temp_filtered = filtered[t[k]:t[k]+2]
        grad = int(np.gradient(temp_filtered)[0]*gradient_scaling)

        # 
        if grad>0:
            tt.append(np.ones(grad))
            x.append(np.ones(grad)*locs[t[k],0])
            y.append(np.ones(grad)*locs[t[k],1])

    tt = np.hstack(tt)
    x = np.hstack(x)
    y = np.hstack(y)
    
    return tt,x,y

#
def get_rms_and_place_field(cell_id,
                            upphases,
                            filtered_Fs,
                            locs,
                            occ_map,
                            arena_size,
                            sigma = 1.0,
                            scale_flag=False,
                            scale_value =1
                            #limits
                           ):

    #
    upphase = upphases[cell_id]
    filtered = filtered_Fs[cell_id]

    # find times when cell is spking and just feed those into the rm 
    
    # get the 
    #t, x, y = get_t_exp_from_gradient(upphase, 
    #                                  filtered,
    #                                  locs)
    
    t, x, y = get_t_exp_from_filtered_F(upphase, 
                                      filtered,
                                      locs)    
    
    #t_exp = get_t_exp_from_filtered_F(upphase, filtered)

    
    # if no spikes are found during movement
    if len(t)==0:
        
        temp = np.zeros((32,32))
        print (" no spiking found... cell: ", cell_id)
        
        return temp,temp,temp,temp
        

    # TODO: not clear what this array is supposed to do ...
    # it seems to take in boolean data?  not super clear
    
    # make the spikes_tracking vstack;  not completely clear this is correct; X->Y inverstion etc? not clear
    spikes_tracking = np.vstack((t,x,y))

    #
    limits = [0,80,0,80]

    # print ("# of spikes: ", spikes_tracking.shape)
    rm = op.analysis.rate_map(occ_map, 
                              spikes_tracking, 
                              bin_edges=bin_edges, 
                              arena_size=arena_size, 
                              #    limits=limits
                             )

    #
    res = op.analysis.rate_map_stats(rm, 
                                     occ_map, 
                                     debug=False)
        
    #
    if True:
        #sigma = 1
        rms = op.general.smooth(rm, 
                            sigma)
    else:
        rms = rm.copy()


    #g
    if False:
        rms = rms.filled(fill_value=0)
        occ_map = occ_map.filled(fill_value=0.001)

    if False:
        rms = rms*scale_value
        #print("scaling")
        #rms = rms+1
    #
    init_thresh = 0.95    
    min_bins = 5
    min_peak = 0.100  #100 μHz
    min_mean = 0.100000   # 100 mHz

    #
    fields, fields_map = op.analysis.place_field(rms, 
                                                init_thresh = init_thresh,
                                                min_bins = min_bins,
                                                #min_peak = min_peak ,
                                                #min_mean = min_mean,
                                                 search_method='sep',
                                              #    limits=limits
                                                )
    #if True:
    #    rms = rms.filled(fill_value=0)

    #
    #print (cell_id, "field ", fields)
    #print ("fields map: ", fields_map)
    
    #
    return rm, rms, fields_map, occ_map
    
#
# def load_locs_traces_running(fname_locs,
#                              fname_traces,
#                              arena_size,
#                              n_frames_per_sec=20,
#                              n_pixels_per_cm=15,   # not used
#                              min_vel=4):          #minimum velocity in cm/sec
#
#
#
#     ####################### LOAD SPIKES ########################
#     data = np.load(fname_traces,
#                    allow_pickle=True)
#
#     #
#     upphases = data['F_upphase']
#     #print ("traces: ", traces.shape, traces[0][:100])
#
#     filtered_Fs = data['F_filtered']
#
#
#     ####################### LOAD LOCATIONS ###################
#     locs = np.load(fname_locs)
#     print ("locs: ", locs.shape)
#
#     #################### COMPUTE VELOCITY ####################
#
#     dists = np.linalg.norm(locs[1:,:]-locs[:-1,:], axis=1)
#     print ("dists: ", dists.shape)
#
#     #
#     vel_all = (dists)*(n_frames_per_sec)
#
#     #
#     from scipy.signal import savgol_filter
#
#     vel_all = savgol_filter(vel_all, n_frames_per_sec, 2)
#
#     #
#     idx_stationary = np.where(vel_all<min_vel)[0]
#     vel = vel_all.copy()
#     vel[idx_stationary] = np.nan
#
#
#
#     ####################### NORMALIZE SIZE OF ARENA  ###########################
#     #
#     min_x = np.min(locs[:,0])
#     max_x = np.max(locs[:,0])
#
#     min_y = np.min(locs[:,1])
#     max_y = np.max(locs[:,1])
#
#     #
#     locs[:,0] = (locs[:,0]-min_x)/(max_x-min_x)*arena_size[0]
#     locs[:,1] = (locs[:,1]-min_y)/(max_y-min_y)*arena_size[1]
#
#     ####################### DELETE EXTRA IMAGING TIME ###########################
#     rec_duration = locs.shape[0]
#
#     upphases = upphases[:,:rec_duration]
#     filtered_Fs = filtered_Fs[:,:rec_duration]
#
#
#     ####################### REMOVE STATIONARY PERIODS ###########################
#
#     #times = np.delete(times, idx_stationary, axis=0)
#
#     locs = np.delete(locs, idx_stationary, axis=0)
#
#     upphases = np.delete(upphases, idx_stationary, axis=1)
#
#     filtered_Fs = np.delete(filtered_Fs, idx_stationary, axis=1)
#
#     ################### COMPUTE TIMES BASED ON THE MOVING PERIODS ######################
#     #
#     times = np.arange(locs.shape[0])
#
#     #
#     print ("Locs: ", locs.shape, " uphases: ", upphases.shape)
#
#     #
#     return locs, upphases, times, filtered_Fs

#
def calc_tuningmap(occupancy, x_edges, y_edges, signaltracking, params):
    '''
    Calculate tuningmap
    Parameters
    ----------
    occupancy : masked np.array
        Smoothed occupancy. Masked where occupancy low
    x_edges : np.array
        Bin edges in x 
    y_edges : np.array
        Bin edges in y
    signaltracking : dict
        # Added by Horst 10-17-2022
          keys:
          signal       # Signal (events or calcium) amplitudes
          x_pos_signal # Tracking x position for signal 
          y_pos_signal # Tracking y position for signal 
        
    params : dict
        MapParams table entry
    
    Returns
    -------
    tuningmap_dict : dict
        - binned_raw : np.array: Binned raw (unsmoothed) signal
        - tuningmap_raw: np masked array: Unsmoothed tuningmap (mask where occupancy low)
        - tuningmap    : np masked array: Smoothed tuningmap (mask where occupancy low)
        - bin_max    : tuple   : (x,y) coordinate of bin with maximum signal
        - max        : float : Max of signal 
        
    '''
    tuningmap_dict = {}
    
    binned_signal = np.zeros_like(occupancy.data)
    # Add one at end to not miss signal at borders
    x_edges[-1] += 1
    y_edges[-1] += 1

    # Look up signal per bin
    for no_x in range(len(x_edges)-1):
        for no_y in range(len(y_edges)-1):
            boolean_x = (signaltracking['x_pos_signal'] >= x_edges[no_x]) & (signaltracking['x_pos_signal'] < x_edges[no_x+1])
            boolean_y = (signaltracking['y_pos_signal'] >= y_edges[no_y]) & (signaltracking['y_pos_signal'] < y_edges[no_y+1])
            extracted_signal = signaltracking['signal'][boolean_x & boolean_y]
            binned_signal[no_y, no_x] = np.nansum(extracted_signal)

    tuningmap_dict['binned_raw'] = binned_signal
    binned_signal = np.ma.masked_where(occupancy.mask, binned_signal)  # Masking. This step is probably unnecessary
    tuningmap_dict['tuningmap_raw'] = binned_signal / occupancy
    
    # Instead of smoothing the raw binned events, substitute those values that are masked in
    # occupancy map with nans.
    # Then use astropy.convolve to smooth padded version of the spikemap 
        
    binned_signal[occupancy.mask] = np.nan
    kernel = Gaussian2DKernel(x_stddev=params['sigma_signal'])

    pad_width = int(5*params['sigma_signal'])
    binned_signal_padded = np.pad(binned_signal, pad_width=pad_width, mode='symmetric')  # as in BNT
    binned_signal_smoothed = astropy.convolution.convolve(binned_signal_padded, kernel, boundary='extend')[pad_width:-pad_width, pad_width:-pad_width]
    binned_signal_smoothed = np.ma.masked_where(occupancy.mask, binned_signal_smoothed)  # Masking. This step is probably unnecessary
    masked_tuningmap = binned_signal_smoothed / occupancy

    tuningmap_dict['tuningmap']       = masked_tuningmap
    tuningmap_dict['bin_max']         = np.unravel_index(masked_tuningmap.argmax(), masked_tuningmap.shape)
    tuningmap_dict['max']             = np.max(masked_tuningmap)
    
    return tuningmap_dict

  
    
from astropy.convolution import Gaussian2DKernel

#
def get_rms_and_place_field_from_tunning_map(cell_id,
                                            upphases,
                                            filtered_Fs,
                                            locs,
                                            occ_map,
                                            arena_size,
                                            sigma = 1.0,
                                            scale_flag=False,
                                            scale_value =1
                                            #limits
                                           ):

    #################################################
    #################################################
    #################################################
    upphase = upphases[cell_id]
    filtered = filtered_Fs[cell_id]
    
    # detect moving periods

    # 
    signaltracking_entry = {"x_pos_signal": locs[:,0],
                            "y_pos_signal": locs[:,1],
                            "signal": upphase*filtered
                           }
    
    # 
    params = {}
    params['sigma_signal'] = 2
    
    # Calculate tuningmap
    tuningmap_dict = calc_tuningmap(occ_map, 
                                    x_edges, 
                                    y_edges, 
                                    signaltracking_entry, 
                                    params)
    
    #
    rm = tuningmap_dict['tuningmap']
    
    #
    res = op.analysis.rate_map_stats(rm, 
                                     occ_map, 
                                     debug=False)
        
    #
    if True:
        #sigma = 1
        rms = op.general.smooth(rm, 
                            sigma)
    else:
        rms = rm.copy()


#     #g
#     if False:
#         rms = rms.filled(fill_value=0)
#         occ_map = occ_map.filled(fill_value=0.001)

#     if False:
#         rms = rms*scale_value

    #
    init_thresh = 0.75    
    min_bins = 5
    min_mean = rm.max()*0.1               # 
    min_peak = rm.max()*0.0001            # 

    #
    fields, fields_map = op.analysis.place_field(rms, 
                                                min_bins = min_bins,
                                                min_peak = min_peak,
                                                min_mean = min_mean,
                                                init_thresh = init_thresh,
                                                search_method='sep',
                                                debug=False
                                              #    limits=limits
                                                )

    
    #
    return rm, rms, fields_map, occ_map



def process_ephys(fname_locs,
                 fname_spikes,
                 fname_locs_times,
                 fname_shift,
                 arena_size,
                 bin_width,
                 n_frames_per_sec=30,     # approximate video frame rate for smoothing velocity...
                 n_pixels_per_cm=15,      # used to compute velocity in cm from pixels
                 min_vel=4):              # minimum velocity in cm/sec


    ####################### LOAD LOCATIONS ###################
    locs = np.load(fname_locs)
    print ("locs: ", locs.shape)
    locs_times = np.load(fname_locs_times)
    locs_times = locs_times-locs_times[0]
    print ("locs tims: ", locs_times.shape)
    print ("locs times: ", locs_times)
    
    
    #################### COMPUTE VELOCITY ####################
    delta_dists = np.linalg.norm(locs[1:,:]-locs[:-1,:], axis=1)
    delta_times = locs_times[1:]-locs_times[:-1]
    print ("distances: ", delta_dists.shape)
    
    #
    vel_all = (delta_dists/delta_times)/n_pixels_per_cm
    print ("vel_all: ", vel_all)
    
    #

    vel_all = savgol_filter(vel_all, 
                            n_frames_per_sec, 
                            2)

    ####################### NORMALIZE SIZE OF ARENA  ###########################
    #
    min_x = np.min(locs[:,0])
    max_x = np.max(locs[:,0])

    min_y = np.min(locs[:,1])
    max_y = np.max(locs[:,1])

    #
    locs[:,0] = (locs[:,0]-min_x)/(max_x-min_x)*arena_size[0]
    locs[:,1] = (locs[:,1]-min_y)/(max_y-min_y)*arena_size[1]

    ####################### REMOVE STATIONARY PERIODS ###########################
    # Not sure this is required for Opexebo to work
    idx_stationary = np.where(vel_all<min_vel)[0]
    vel = vel_all.copy()
    vel[idx_stationary] = np.nan
   
    #
    locs = np.delete(locs, idx_stationary, axis=0)
    locs_binned = locs/bin_width
    locs_times = np.delete(locs_times, idx_stationary, axis=0)

    ############### LOAD EPHYS ##################
    spikes = np.load(fname_spikes).astype('float32')
    print ("loaded spikes: ", spikes)

    # sort by time
    idx = np.argsort(spikes[:,0])
    spikes = spikes[idx]

    # convert to seconds
    sample_rate = 30000
    spikes[:,0] = spikes[:,0]/sample_rate

    # remove spikes occuring before video is turned on
    shift = np.load(fname_shift)[0]
    print ("time shift (sec): ", shift)

    #
    idx = np.where(spikes[:,0]>=shift)[0]
    print ('spikes pre: ', spikes.shape)

    spikes = spikes[idx[0]:]
    spikes[:,0]-= shift
    print ('spikes post: ', spikes.shape)

    ################### COMPUTE TIMES BASED ON THE MOVING PERIODS ######################
    #
    #times_locs = np.arange(locs.shape[0])
       


    #
    return locs, locs_binned, locs_times, spikes


#
def load_locs_traces(fname,
                     arena_size):
    #
    locs = np.load(fname)
    print(locs.shape)

    ####################### COMPUTE SPATIAL OCCUPANCY ###########################
    times = np.arange(locs.shape[0])

    #
    min_x = np.min(locs[:, 0])
    max_x = np.max(locs[:, 0])

    min_y = np.min(locs[:, 1])
    max_y = np.max(locs[:, 1])

    #
    locs[:, 0] = (locs[:, 0] - min_x) / (max_x - min_x) * arena_size[0]
    locs[:, 1] = (locs[:, 1] - min_y) / (max_y - min_y) * arena_size[1]

    ####################### LOAD SPIKES ########################
    fname = '/media/cat/4TB1/donato/nathalie/DON-007050/FS1/binarized_traces.npz'
    bin_ = np.load(fname,
                   allow_pickle=True)

    #
    upphases = bin_['F_upphase']
    # print ("traces: ", traces.shape, traces[0][:100])

    filtered_Fs = bin_['F_filtered']

    #
    return locs, upphases, times, filtered_Fs


#
def get_t_exp_from_filtered_F(upphase,
                              filtered,
                              locs):
    ''' function uses upphase detected spiking
        and then uses the value of the filtered [ca]
    '''

    # check when upphase spiking occurs
    t = np.where(upphase > 0)[0]

    print("locs: ", locs.shape)

    tt = []
    x = []
    y = []
    gradient_scaling = 20
    for k in range(t.shape[0]):

        #
        temp_filtered = filtered[t[k]]
        grad = int(temp_filtered)

        #
        if grad > 0:
            tt.append(np.ones(grad))
            x.append(np.ones(grad) * locs[t[k], 0])
            y.append(np.ones(grad) * locs[t[k], 1])

    tt = np.hstack(tt)
    x = np.hstack(x)
    y = np.hstack(y)

    return tt, x, y


#
def get_t_exp_from_gradient(
        upphase,
        filtered,
        locs
):
    ''' function uses upphase detected spiking
        and then scales the spiking based on the gradient of the filtered [ca]
    '''

    t = np.where(upphase > 0)[0]

    print("locs: ", locs.shape)

    tt = []
    x = []
    y = []
    gradient_scaling = 20
    for k in range(t.shape[0]):

        #
        temp_filtered = filtered[t[k]:t[k] + 2]
        grad = int(np.gradient(temp_filtered)[0] * gradient_scaling)

        #
        if grad > 0:
            tt.append(np.ones(grad))
            x.append(np.ones(grad) * locs[t[k], 0])
            y.append(np.ones(grad) * locs[t[k], 1])

    tt = np.hstack(tt)
    x = np.hstack(x)
    y = np.hstack(y)

    return tt, x, y


#
def get_rms_and_place_field(cell_id,
                            upphases,
                            filtered_Fs,
                            locs,
                            occ_map,
                            arena_size,
                            sigma=1.0,
                            scale_flag=False,
                            scale_value=1
                            # limits
                            ):
    #
    upphase = upphases[cell_id]
    filtered = filtered_Fs[cell_id]

    # find times when cell is spking and just feed those into the rm

    # get the
    # t, x, y = get_t_exp_from_gradient(upphase,
    #                                  filtered,
    #                                  locs)

    t, x, y = get_t_exp_from_filtered_F(upphase,
                                        filtered,
                                        locs)

    # t_exp = get_t_exp_from_filtered_F(upphase, filtered)

    # if no spikes are found during movement
    if len(t) == 0:
        temp = np.zeros((32, 32))
        print(" no spiking found... cell: ", cell_id)

        return temp, temp, temp, temp

    # TODO: not clear what this array is supposed to do ...
    # it seems to take in boolean data?  not super clear

    # make the spikes_tracking vstack;  not completely clear this is correct; X->Y inverstion etc? not clear
    spikes_tracking = np.vstack((t, x, y))

    #
    limits = [0, 80, 0, 80]

    # print ("# of spikes: ", spikes_tracking.shape)
    rm = op.analysis.rate_map(occ_map,
                              spikes_tracking,
                              bin_edges=bin_edges,
                              arena_size=arena_size,
                              #    limits=limits
                              )

    #
    res = op.analysis.rate_map_stats(rm,
                                     occ_map,
                                     debug=False)

    #
    if True:
        # sigma = 1
        rms = op.general.smooth(rm,
                                sigma)
    else:
        rms = rm.copy()

    # g
    if False:
        rms = rms.filled(fill_value=0)
        occ_map = occ_map.filled(fill_value=0.001)

    if False:
        rms = rms * scale_value
        # print("scaling")
        # rms = rms+1
    #
    init_thresh = 0.95
    min_bins = 5
    min_peak = 0.100  # 100 μHz
    min_mean = 0.100000  # 100 mHz

    #
    fields, fields_map = op.analysis.place_field(rms,
                                                 init_thresh=init_thresh,
                                                 min_bins=min_bins,
                                                 # min_peak = min_peak ,
                                                 # min_mean = min_mean,
                                                 search_method='sep',
                                                 #    limits=limits
                                                 )
    # if True:
    #    rms = rms.filled(fill_value=0)

    #
    # print (cell_id, "field ", fields)
    # print ("fields map: ", fields_map)

    #
    return rm, rms, fields_map, occ_map


#
def compute_overlap(fields_map):
    #
    max_size = 0
    for k in range(len(fields_map)):
        idx = np.where(fields_map[k] > 0)
        fields_map[k][idx] = 1

        #
        idx2 = np.sum(fields_map[k])
        if idx2 > max_size:
            max_size = idx2

    #
    idx = np.logical_and(fields_map[0],
                         fields_map[1],
                         ).sum()

    #
    overlap = idx / max_size

    #
    return overlap


#
def load_locs_traces_running(fname_locs,
                             fname_traces,
                             arena_size,
                             n_frames_per_sec=20,
                             n_pixels_per_cm=15,  # not used
                             min_vel=4):  # minimum velocity in cm/sec

    #print ('')
    #################### LOAD THE DATA OFFSET ##################
    try:
        start = np.loadtxt(os.path.split(fname_locs)[0]+"/start.txt", dtype=np.int)
    except:
        start = 0
    print ("OFFSET is ", start)


    ####################### LOAD SPIKES ########################
    data = np.load(fname_traces,
                   allow_pickle=True)

    #
    upphases = data['F_upphase']
    #print ("upphases : ", upphases.shape)

    filtered_Fs = data['F_filtered']

    ####################### LOAD LOCATIONS ###################
    locs = np.load(fname_locs)
    locs = locs[start:]
    #print("locs after trimming: ", locs.shape)

    #################### COMPUTE VELOCITY ####################

    dists = np.linalg.norm(locs[1:, :] - locs[:-1, :], axis=1)

    #
    vel_all = (dists) * (n_frames_per_sec)

    #

    vel_all = savgol_filter(vel_all, n_frames_per_sec, 2)

    #
    idx_stationary = np.where(vel_all < min_vel)[0]
    vel = vel_all.copy()
    vel[idx_stationary] = np.nan

    ####################### NORMALIZE SIZE OF ARENA  ###########################
    #
    min_x = np.min(locs[:, 0])
    max_x = np.max(locs[:, 0])

    min_y = np.min(locs[:, 1])
    max_y = np.max(locs[:, 1])

    #
    locs[:, 0] = (locs[:, 0] - min_x) / (max_x - min_x) * arena_size[0]
    locs[:, 1] = (locs[:, 1] - min_y) / (max_y - min_y) * arena_size[1]

    ####################### REMOVE STATIONARY PERIODS ###########################

    # times = np.delete(times, idx_stationary, axis=0)

    locs = np.delete(locs, idx_stationary, axis=0)

    upphases = np.delete(upphases, idx_stationary, axis=1)

    filtered_Fs = np.delete(filtered_Fs, idx_stationary, axis=1)

    ####################### DELETE EXTRA IMAGING TIME ###########################
    rec_duration = locs.shape[0]

    upphases = upphases[:, :rec_duration]
    filtered_Fs = filtered_Fs[:, :rec_duration]

    ################### COMPUTE TIMES BASED ON THE MOVING PERIODS ######################
    #
    times = np.arange(locs.shape[0])

    #
    fps = 20
    print("Duration of moving periods ", locs.shape[0]/fps, " # cells: ", upphases.shape[0])

    #
    return locs, upphases, times, filtered_Fs


#

def calc_tuningmap(occupancy, x_edges, y_edges, signaltracking, params):
    '''
    Calculate tuningmap
    Parameters
    ----------
    occupancy : masked np.array
        Smoothed occupancy. Masked where occupancy low
    x_edges : np.array
        Bin edges in x
    y_edges : np.array
        Bin edges in y
    signaltracking : dict
        # Added by Horst 10-17-2022
          keys:
          signal       # Signal (events or calcium) amplitudes
          x_pos_signal # Tracking x position for signal
          y_pos_signal # Tracking y position for signal

    params : dict
        MapParams table entry

    Returns
    -------
    tuningmap_dict : dict
        - binned_raw : np.array: Binned raw (unsmoothed) signal
        - tuningmap_raw: np masked array: Unsmoothed tuningmap (mask where occupancy low)
        - tuningmap    : np masked array: Smoothed tuningmap (mask where occupancy low)
        - bin_max    : tuple   : (x,y) coordinate of bin with maximum signal
        - max        : float : Max of signal

    '''
    tuningmap_dict = {}

    binned_signal = np.zeros_like(occupancy.data)
    # Add one at end to not miss signal at borders
    x_edges[-1] += 1
    y_edges[-1] += 1

    # Look up signal per bin
    for no_x in range(len(x_edges) - 1):
        for no_y in range(len(y_edges) - 1):
            boolean_x = (signaltracking['x_pos_signal'] >= x_edges[no_x]) & (
                        signaltracking['x_pos_signal'] < x_edges[no_x + 1])
            boolean_y = (signaltracking['y_pos_signal'] >= y_edges[no_y]) & (
                        signaltracking['y_pos_signal'] < y_edges[no_y + 1])
            extracted_signal = signaltracking['signal'][boolean_x & boolean_y]
            binned_signal[no_y, no_x] = np.nansum(extracted_signal)

    tuningmap_dict['binned_raw'] = binned_signal
    binned_signal = np.ma.masked_where(occupancy.mask, binned_signal)  # Masking. This step is probably unnecessary
    tuningmap_dict['tuningmap_raw'] = binned_signal / occupancy

    # Instead of smoothing the raw binned events, substitute those values that are masked in
    # occupancy map with nans.
    # Then use astropy.convolve to smooth padded version of the spikemap

    binned_signal[occupancy.mask] = np.nan
    kernel = Gaussian2DKernel(x_stddev=params['sigma_signal'])

    pad_width = int(5 * params['sigma_signal'])
    binned_signal_padded = np.pad(binned_signal, pad_width=pad_width, mode='symmetric')  # as in BNT
    binned_signal_smoothed = astropy.convolution.convolve(binned_signal_padded, kernel, boundary='extend')[
                             pad_width:-pad_width, pad_width:-pad_width]
    binned_signal_smoothed = np.ma.masked_where(occupancy.mask,
                                                binned_signal_smoothed)  # Masking. This step is probably unnecessary
    masked_tuningmap = binned_signal_smoothed / occupancy

    tuningmap_dict['tuningmap'] = masked_tuningmap
    tuningmap_dict['bin_max'] = np.unravel_index(masked_tuningmap.argmax(), masked_tuningmap.shape)
    tuningmap_dict['max'] = np.max(masked_tuningmap)

    return tuningmap_dict


#
def get_rms_and_place_field_from_tunning_map(cell_id,
                                             upphases,
                                             filtered_Fs,
                                             locs,
                                             occ_map,
                                             arena_size,
                                             sigma=1.0,
                                             scale_flag=False,
                                             scale_value=1
                                             # limits
                                             ):
    #################################################
    #################################################
    #################################################
    upphase = upphases[cell_id]
    filtered = filtered_Fs[cell_id]

    # detect moving periods

    #
    signaltracking_entry = {"x_pos_signal": locs[:, 0],
                            "y_pos_signal": locs[:, 1],
                            "signal": upphase * filtered
                            }

    #
    params = {}
    params['sigma_signal'] = 2

    # Calculate tuningmap
    tuningmap_dict = calc_tuningmap(occ_map,
                                    x_edges,
                                    y_edges,
                                    signaltracking_entry,
                                    params)

    #
    rm = tuningmap_dict['tuningmap']

    #
    res = op.analysis.rate_map_stats(rm,
                                     occ_map,
                                     debug=False)

    #
    if True:
        # sigma = 1
        rms = op.general.smooth(rm,
                                sigma)
    else:
        rms = rm.copy()

    #
    init_thresh = 0.75
    min_bins = 5
    min_mean = rm.max() * 0.1  #
    min_peak = rm.max() * 0.0001  #

    #
    fields, fields_map = op.analysis.place_field(rms,
                                                 min_bins=min_bins,
                                                 min_peak=min_peak,
                                                 min_mean=min_mean,
                                                 init_thresh=init_thresh,
                                                 search_method='sep',
                                                 debug=False
                                                 #    limits=limits
                                                 )

    #
    return rm, rms, fields_map, occ_map


#
def get_rms_and_place_field_from_tunning_map_split_test(cell_id,
                                                        upphases,
                                                        filtered_Fs,
                                                        locs,
                                                        occ_map,
                                                        arena_size,
                                                        x_edges,
                                                        y_edges,
                                                        sigma=1.0,
                                                        circular_shuffle=False,
                                                        split = True,
                                                        # limits
                                                        ):
    #################################################
    #################################################
    #################################################

    # these contain only data for moving periods
    upphase = upphases[cell_id]
    filtered = filtered_Fs[cell_id]

    # if shuffle:
    if circular_shuffle:
        # temp = np.random.choice(np.arange(1000,upphase.shape[0]-1000))
        temp = np.random.choice(np.arange(upphase.shape[0] // 5, upphase.shape[0] // 5 * 4))
        upphase = np.roll(upphase, temp)
        filtered = np.roll(filtered, temp)

    # split data or NOT depending on flag
    idxs = []
    if split:
        # Option 1: first half and second half
        if True:
            idxs.append(np.arange(0, upphase.shape[0] // 2, 1))
            idxs.append(np.arange(upphase.shape[0] // 2, upphase.shape[0], 1))

        # Option 2: 1 min chunks
        else:
            fps = 20
            n_sec = 60
            idx_temp = np.split(np.arange(upphase.shape[0]), np.arange(0, upphase.shape[0], fps * n_sec), axis=0)
            # print ("idx temp: ", idx_temp)
            idxs.append(np.hstack(idx_temp[::2]))
            idxs.append(np.hstack(idx_temp[1::2]))

    else:
        idxs.append(np.arange(upphase.shape[0]))

    #
    fields = []
    fields_map = []
    rms = []
    for k in range(len(idxs)):
        #
        signaltracking_entry = {"x_pos_signal": locs[idxs[k], 0],
                                "y_pos_signal": locs[idxs[k], 1],
                                "signal": upphase[idxs[k]] * filtered[idxs[k]]
                                }

        #
        params = {}
        params['sigma_signal'] = 2

        # Calculate tuningmap
        tuningmap_dict = calc_tuningmap(occ_map,
                                        x_edges,
                                        y_edges,
                                        signaltracking_entry,
                                        params)

        #
        rm = tuningmap_dict['tuningmap']

        #
        rm = op.general.smooth(rm,
                               sigma)

        #
        # init_thresh = 0.75
        init_thresh = 0.98
        min_bins = 5
        min_mean = rm.max() * 0.1  #
        min_peak = rm.max() * 0.001  #

        #
        field, field_map = op.analysis.place_field(rm,
                                                   min_bins=min_bins,
                                                   min_peak=min_peak,
                                                   min_mean=min_mean,
                                                   init_thresh=init_thresh,
                                                   search_method='sep',
                                                   debug=False
                                                   #    limits=limits
                                                   )
        rms.append(rm)
        fields.append(field)
        fields_map.append(field_map)

    #
    return rms, fields, fields_map


def plot_cell_3d(ax,
                 rms,
                 zmax=None):
    x = []
    y = []
    z = []

    x = np.arange(rms.shape[0])
    y = np.arange(rms.shape[1])
    x, y = np.meshgrid(x, y)

    #     #
    for k in range(rms.shape[0]):
        for p in range(rms.shape[1]):
            z.append(rms[k, p])

    z = np.hstack(z).reshape(rms.shape[0], rms.shape[1])

    ax.contour3D(x, y, z, 300, cmap='viridis')

    if zmax is not None:
        ax.set_zlim(0, zmax)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z');

#
def plot_fields(cell,
                occ_map):

    fig = plt.figure(figsize=(15,10))

    cell_id = cell['cell_id']
    fields_map = cell["fields_map_split"]
    rms = cell["rms_split"]
    rms_all = cell['rms_all'][0]
    spatial_info_zscores = cell["spatial_info_zscores"]
    #print ("SI zscores: ", spatial_info_zscores)
    spatial_info = cell['spatial_info']



    #print ("sI zscores: ", len(spatial_info_zscores))
    #print ("sI: ", len(spatial_info))


    #
    fields_map_all = cell['fields_map_all'][0]#.squeeze()

    # plot all data
    ax = fig.add_subplot(231, projection='3d')
    ax.azim = 250
    ax.dist = 8
    ax.elev = 30

    #
    plot_cell_3d(ax, rms_all)
    ax.zaxis.set_rotate_label(False)  # disable automatic rotation


    ax.set_zlabel('Smooth Rate Map', fontsize=10
                  , rotation=90
                  )

    z_max = np.max(rms_all)

    #
    plt.title("all (moving) times, SI Zscore: "+str(round(spatial_info_zscores[0],2)), fontsize=10, pad=0.9)

    #
    res = op.analysis.rate_map_stats(rms_all,
                                     occ_map,
                                     debug=False)

    #
    coh = op.analysis.rate_map_coherence(rms_all)

    #
    text = "SI_rate: " + str(round(res['spatial_information_rate'], 2)) + \
           "  SI_cont: " + str(round(res['spatial_information_content'], 2)) + \
           "  Sparse: " + str(round(res['sparsity'], 2)) + ' \n ' + \
           "Select: " + str(round(res['selectivity'], 2)) + \
           "  Peak_r: " + str(round(res['peak_rate'], 2)) + \
           "  Mean_r: " + str(round(res['mean_rate'], 2)) + \
           "  Coh: " + str(round(coh, 2))

    #
    ax2 = plt.subplot(2, 3, 4)
    plt.imshow(fields_map_all,
               extent=[0,fields_map_all.shape[0], fields_map_all.shape[1],0])
    plt.xlim(0,fields_map_all.shape[0])
    plt.ylim(0,fields_map_all.shape[1])

    #
    plt.ylabel("Opexebo place fields")
    plt.suptitle("cell " + str(cell_id) + "\n" + str(text), fontsize=10)

    # plot the split data
    texts = ['first half', 'second half']
    for k in range(2):
        ax2 = fig.add_subplot(2,3,k+2, projection='3d')
        ax2.azim = 250
        ax2.dist = 8
        ax2.elev = 30
        #
        plot_cell_3d(ax2,
                     rms[k],
                     z_max)

        ax2.set_xlabel('')
        ax2.set_ylabel('')
        ax2.set_zlabel('')

        #
        plt.title(texts[k] + ", SI Zscore: " + str(round(spatial_info_zscores[k+1],2)), fontsize=10, pad=0.9)

        #
        ax2 = plt.subplot(2, 3, k + 5)
        plt.xlim(0, fields_map[k].shape[0])
        plt.ylim(0, fields_map[k].shape[1])
        plt.imshow(fields_map[k],
                   extent=[0, fields_map_all.shape[0], fields_map_all.shape[1], 0])

    #
def compute_all_place_fields_parallel(n_tests,
                                      root_dir,
                                      animal_id,
                                      session_id,
                                      parallel=True,
                                      cell_ids=None,
                                      ):

    fname_temp = os.path.join(root_dir,
                              animal_id,
                              session_id,
                              '*locs.npy')
    
    #
    fname_locs = glob.glob(fname_temp)[0]

    #
    fname_test= os.path.split(fname_locs)[0] + '/*binarized_traces*'
    fname_traces = glob.glob(fname_test)[0]
    if os.path.exists(fname_traces)==False:
        print ("traces not found: ", fname_test)
        return None
    
    
    #########################################################
    ########## ARENA AND OCCUPANCY MAP COMPUTATIONS #########
    #########################################################
    arena_x = [300, 1550]
    arena_y = [175, 1400]
    arena_size = [80, 80]
    arena_shape = 'square'
    bin_width = 2.5

    #
    locs, upphases, times, filtered_Fs = load_locs_traces_running(fname_locs,
                                                                  fname_traces,
                                                                  arena_size)

    # compute spatial occpuancy map; only requires the locatoins and
    occ_map, coverage, bin_edges = op.analysis.spatial_occupancy(times,
                                                                 locs.T,
                                                                 arena_size,
                                                                 bin_width=bin_width)

    #
    x_edges = bin_edges[0]
    y_edges = bin_edges[1]

    #
    fname_occ_map = os.path.join(os.path.split(fname_traces)[0], "occ_map.npy")

    #
    occ_map.dump(fname_occ_map)

    #########################################################
    ############### PLACE FIELD COMPUTATION #################
    #########################################################
    if cell_ids is None:
        cell_ids = np.arange(0, upphases.shape[0], 1)

    #
    sigma = 1.5
    circular_shuffle = False

    #
    overlaps = []
    s_info = []
    res_arrays = []
    cohs_array = []
    D_array = []

    # make root directory for data saving
    root_dir = os.path.join(os.path.split(fname_traces)[0],
                            "cell_analysis")
    try:
        os.mkdir(root_dir)
    except:
        pass

    #
    if parallel:
        parmap.map(check_cell_id_field,
               cell_ids,
               root_dir,
               upphases,
               filtered_Fs,
               locs,
               occ_map,
               arena_size,
               sigma,
               circular_shuffle,
               n_tests,
               x_edges,
               y_edges,
               pm_pbar = True,
               )
    else:
        for cell_id in cell_ids:
            check_cell_id_field(cell_id,
                                root_dir,
                                upphases,
                                filtered_Fs,
                                locs,
                                occ_map,
                                arena_size,
                                sigma,
                                circular_shuffle,
                                n_tests,
                                x_edges,
                                y_edges,
                                )


#
def check_cell_id_field(cell_id,
                        root_dir,
                        upphases,
                        filtered_Fs,
                        locs,
                        occ_map,
                        arena_size,
                        sigma,
                        circular_shuffle,
                        n_tests,
                        x_edges,
                        y_edges
                        ):

    #
    fname_out = os.path.join(root_dir, str(cell_id)+'.npy')
    if os.path.exists(fname_out):
        return


    #
    D = {}

    ###############################################
    ########## GET RMS FOR WHOLE DATA #############
    ###############################################
    split = False  # do not split data
    rms_all, _, fields_map_all = get_rms_and_place_field_from_tunning_map_split_test(cell_id,
                                                                                       upphases,
                                                                                       filtered_Fs,
                                                                                       locs,
                                                                                       occ_map,
                                                                                       arena_size,
                                                                                       x_edges,
                                                                                       y_edges,
                                                                                       sigma,
                                                                                       circular_shuffle,
                                                                                       split,
                                                                                       )


    ###############################################
    ########## GET RMS FOR SPLIT DATA #############
    ###############################################
    split = True
    rms_split, _, fields_map_split = get_rms_and_place_field_from_tunning_map_split_test(cell_id,
                                                                                       upphases,
                                                                                       filtered_Fs,
                                                                                       locs,
                                                                                       occ_map,
                                                                                       arena_size,
                                                                                       x_edges,
                                                                                       y_edges,
                                                                                       sigma,
                                                                                       circular_shuffle,
                                                                                       split
                                                                                       )



    ###############################################
    ###############################################
    ###############################################
    # run the shuffle condition 100 times
    rms_shuffle = []
    fields_map_shuffle = []
    circular_shuffle = True
    parallel_flag = False
    cell_ids = np.zeros(n_tests, dtype=np.int32) + cell_id
    split = False

    #
    res = []
    for cell_id in cell_ids:
        res.append(get_rms_and_place_field_from_tunning_map_split_test(cell_id,
                                                                     upphases,
                                                                     filtered_Fs,
                                                                     locs,
                                                                     occ_map,
                                                                     arena_size,
                                                                     x_edges,
                                                                     y_edges,
                                                                     sigma,
                                                                     circular_shuffle,
                                                                     split
                                                                         ))

    #
    for re in res:
        rms_shuffle.append(re[0])
        fields_map_shuffle.append(re[2])

    ###############################################
    ###### COMPUTE SPATIAL INFO OVER ALL DATA #####
    ###############################################
    spatial_infos = []
    res_array = []

    # add the nonsufhlled SI
    res = op.analysis.rate_map_stats(rms_all,
                                     occ_map,
                                     debug=False)
    res_array.append(res)
    spatial_infos.append(res['spatial_information_content'])

    # and the split data
    for p in range(len(rms_split)):
        res = op.analysis.rate_map_stats(rms_split[p],
                                             occ_map,
                                             debug=False)
        res_array.append(res)
        spatial_infos.append(res['spatial_information_content'])

    # and then add all the final data
    for p in range(len(rms_shuffle)):
        res = op.analysis.rate_map_stats(rms_shuffle[p],
                                             occ_map,
                                             debug=False)
        res_array.append(res)
        spatial_infos.append(res['spatial_information_content'])

    spatial_info_zscores = stats.zscore(np.hstack(spatial_infos), nan_policy='omit')

    ###############################################
    ############### COMPUTE OVERLAPS ##############
    ###############################################
    # loop over the 2
    overlaps = []
    overlaps.append(compute_overlap(fields_map_split))

    #####################################################
    ############# COHERENCE #############################
    #####################################################
    cohs = []
    for k in range(2):
        temp = op.analysis.rate_map_coherence(rms_split[k])
        cohs.append(temp)

    #
    D = {}
    D['cell_id'] = cell_id
    D["overlaps"] = overlaps
    D["spatial_info"] = spatial_infos
    D["spatial_info_zscores"] = spatial_info_zscores
    D["coherence"] = cohs
    D["fields_map_split"] = fields_map_split
    D["res_array"] = res_array
    D["rms_split"] = rms_split
    D['rms_all'] = rms_all
    D['fields_map_all'] = fields_map_all

    #
    if False:
        if n_tests > 0:
            print(cell_id, "overlap: ", overlaps[0], ", zscore: ", rms_zscores[0])
            print("       spatial infos: ", spatial_infos[0], ", zscore: ", spatial_info_zscores[0])

    #
    np.save(fname_out, D, allow_pickle=True)



def plot_all_metrics(cells):
    #
    sis = []
    sis_z = []
    overls = []
    overls_z = []
    cohs = []
    spars = []
    si_rates = []
    cell_ids = []
    peak_rs = []
    selectivity = []
    for cell in cells:
        cell_ids.append(cell['cell_id'])
        si = cell['spatial_info'][0]
        si_z = cell['spatial_info_zscores'][0]
        overl = cell['overlaps'][0]

        #
        temp = cell['res_array']
        spar = []
        si_rate = []
        peaks = []
        selects = []
        for k in range(len(temp)):
            spar.append(temp[k]['sparsity'])
            si_rate.append(temp[k]['spatial_information_rate'])
            peaks.append(temp[k]['peak_rate'])
            selects.append(temp[k]['selectivity'])

        #
        spar = np.nanmean(np.hstack((spar)))
        si_rate = np.nanmean(np.hstack((si_rate)))
        peaks = np.nanmean(np.hstack((peaks)))
        selects = np.nanmean(np.hstack((selects)))

        #
        coh = np.nanmean(np.hstack(cell['coherence']))

        #
        sis.append(si)
        sis_z.append(si_z)
        overls.append(overl)
        #overls_z.append(overl_z)
        cohs.append(coh)
        spars.append(spar)
        si_rates.append(si_rate)
        peak_rs.append(peaks)
        selectivity.append(selects)

    #

    plt.figure(figsize=(15, 10))

    ############################
    ax = plt.subplot(3, 3, 1)
    color = 'blue'
    ax.scatter(overls,
               sis,
               s=100,
               # alpha=overl_z,
               edgecolor='black',
               c=color)
    plt.xlabel(" overlap of place field (1st half and 2nd half)")
    plt.ylabel(" spatial info (average 1st half and 2nd half)")
    for k in range(len(overls)):
        ax.annotate(str(cell_ids[k]), (overls[k], sis[k]))

    ############################
    ax = plt.subplot(3, 3, 2)
    color = 'blue'
    #print(overls_z)
    ax.scatter(overls,
               sis_z,
               s=100,
               # alpha=overl_z,
               edgecolor='black',
               c=color)

    #
    plt.xlabel(" overlap of place field (1st half and 2nd half)")
    plt.ylabel(" zscore spatial info")
    for k in range(len(overls)):
        ax.annotate(str(cell_ids[k]), (overls[k], sis_z[k]))

    x = np.arange(0.0, 1, 0.01)
    y1 = np.zeros(x.shape[0])+3
    y2 = np.zeros(x.shape[0])+np.nanmax(sis_z)

    #fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(6, 6))
    ax.fill_between(x, y1, y2, color='grey',alpha=.5)

    ############################
    ax = plt.subplot(3, 3, 3)
    color = 'blue'
    ax.scatter(overls,
               si_rates,
               s=100,
               # alpha=overl_z,
               edgecolor='black',
               c=color)
    plt.xlabel(" overlap of place field (1st half and 2nd half)")
    plt.ylabel(" spatial information rate")
    for k in range(len(overls)):
        ax.annotate(str(cell_ids[k]), (overls[k], si_rates[k]))

    plt.semilogy()

    ############################
    # if False:
    #     ax = plt.subplot(3, 3, 4)
    #     color = 'red'
    #     ax.scatter(overls,
    #                overls_z,
    #                s=100,
    #                # alpha=overl_z,
    #                edgecolor='black',
    #                c=color)
    #     plt.xlabel(" overlap of place field (1st half and 2nd half)")
    #     plt.ylabel(" zscore overlap")
    #     for k in range(len(overls)):
    #         ax.annotate(str(cell_ids[k]), (overls[k], overls_z[k]))

    ############################
    ax = plt.subplot(3, 3, 4)
    color = 'green'
    ax.scatter(overls,
               cohs,
               s=100,
               # alpha=overl_z,
               edgecolor='black',
               c=color)
    plt.xlabel(" overlap of place field (1st half and 2nd half)")
    plt.ylabel(" coherence")
    for k in range(len(overls)):
        ax.annotate(str(cell_ids[k]), (overls[k], cohs[k]))

    ############################
    ax = plt.subplot(3, 3, 5)
    color = 'black'
    ax.scatter(overls,
               spars,
               s=100,
               # alpha=overl_z,
               edgecolor='black',
               c=color)
    plt.xlabel(" overlap of place field (1st half and 2nd half)")
    plt.ylabel(" sparsity")
    for k in range(len(overls)):
        ax.annotate(str(cell_ids[k]), (overls[k], spars[k]))

    ############################
    ax = plt.subplot(3, 3, 6)
    color = 'brown'
    ax.scatter(overls,
               peak_rs,
               s=100,
               # alpha=overl_z,
               edgecolor='black',
               c=color)
    plt.xlabel(" overlap of place field (1st half and 2nd half)")
    plt.ylabel(" peak reates")
    for k in range(len(overls)):
        ax.annotate(str(cell_ids[k]), (overls[k], peak_rs[k]))

    ############################
    ax = plt.subplot(3, 3, 7)
    color = 'darkblue'
    ax.scatter(overls,
               selectivity,
               s=100,
               # alpha=overl_z,
               edgecolor='black',
               c=color)
    plt.xlabel(" overlap of place field (1st half and 2nd half)")
    plt.ylabel(" selectivity")
    for k in range(len(overls)):
        ax.annotate(str(cell_ids[k]), (overls[k], selectivity[k]))




    plt.show()


def find_placefields(root_dir, animal_id, ax=None, label=''):
    #
    sessions = np.loadtxt(root_dir + '/' + animal_id + '/sessions.txt', dtype='str')

    #
    if ax==None:
        fig = plt.figure()
        ax=plt.subplot(111)

    #
    place_fields = []
    n_cells = []
    for ctr, session in tqdm(enumerate(sessions)):

        #
        root_dir = os.path.join(session,
                                'cell_analysis')

        #
        cell_id = 0
        computed_cells = []
        # loop over many cells, just in case there are errors etc in the pipelines
        for k in range(2000):

            cell = load_cell(root_dir,
                             k)

            if cell is not None:
                # print ("found: ", k)
                computed_cells.append(cell)
                cell_id += 1

        #
        threshold = 3.0
        n_si_zscore = 0
        good_cells=[]
        for cell in computed_cells:
            si = cell['spatial_info'][0]
            si_z = cell['spatial_info_zscores'][0]

            if si_z > threshold:
                n_si_zscore += 1
                good_cells.append(cell)


        #

        #
        fname_out = os.path.join(root_dir, animal_id, session, "place_cells.npy")
        np.save(fname_out, good_cells)
        n_cells.append(n_si_zscore)

    ax.plot(np.arange(1,len(n_cells)+1,1), n_cells,
            label=label
                #c='black'
                )
    plt.ylim(0,240)


    plt.title(animal_id)
    plt.ylabel("# of place cells")
    plt.xlabel("FS Day")
    plt.legend()

#
def return_cells(root_dir, animal_id, session_id):
    #
    # fname_locs = ('/media/cat/4TB/donato/nathalie/DON-007050/FS9/DON-007050_20211030_TR-BSL_FS9-ACQDLC_resnet50_open_arena_white_floorSep8shuffle1_200000_locs.npy')
    fname_temp = os.path.join(root_dir,
                              animal_id,
                              session_id,
                              '*locs.npy')
    
    #
    fname_locs = glob.glob(fname_temp)[0]
    #print ("fname_locs: ", os.path.split(fname_locs)[0])

    #
    fname_test= os.path.split(fname_locs)[0] + '/*binarized_traces*'
    fname_traces = glob.glob(fname_test)[0]
    if os.path.exists(fname_traces)==False:
        print ("traces not found: ", fname_test)
        return None
    #
    root_dir = os.path.join(os.path.split(fname_traces)[0],
                            'cell_analysis')
    
    #
    fname_occ_map = os.path.split(root_dir)[0] + '/occ_map.npy'
    if os.path.exists(fname_occ_map):
        occ_map = np.load(fname_occ_map, allow_pickle=True)
    else:

        #arena_x = [300, 1550]
        #arena_y = [175, 1400]
        arena_size = [80, 80]
        #arena_shape = 'square'
        bin_width = 2.5

        #
        locs, _, times, _ = load_locs_traces_running(fname_locs,
                                                    fname_traces,
                                                    arena_size)

        # compute spatial occpuancy map; only requires the locatoins and
        occ_map, _, _ = op.analysis.spatial_occupancy(times,
                                                    locs.T,
                                                    arena_size,
                                                    bin_width=bin_width)

        #
        occ_map.dump(fname_occ_map)


    #
    cell_id = 0
    cells = []
    for k in trange(2000):

        cell = load_cell(root_dir,
                         k)

        if cell is not None:
            # print ("found: ", k)
            cells.append(cell)
            cell_id += 1

    print(" Done... found: ", len(cells), " cells")

    return cells, occ_map



def load_cell(root_dir,
              cell_id):
    try:
        fname = os.path.join(root_dir, str(cell_id) + '.npy')
        cell = np.load(fname, allow_pickle=True)
    except:
        return None

    return cell.item()


def plot_contours(root_dir,
                  animal_id,
                  session_id,
                  std_threshold):

    #
    cells, occ_map = return_cells(root_dir,
                                  animal_id,
                                  session_id)

    #
    plt.figure()
    #threshold = 3
    ctr = 0
    for cell in cells:
        if True:
            if cell["spatial_info_zscores"][0] < std_threshold:
                continue
            ctr += 1

            fields_map_all = cell['fields_map_all'][0] * 10
            img_grey = fields_map_all.astype('uint8')

            thresh = 1
            _, thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            try:
                contours = contours[0].squeeze()
                contours = np.append(contours, contours[0][None], axis=0)
                plt.plot(contours[:, 0], contours[:, 1],
                         linewidth=3)
            except:
                pass
        #except:
        #    pass

    plt.suptitle(animal_id + " " + session_id + " # place cells: " + str(ctr))
    plt.show()