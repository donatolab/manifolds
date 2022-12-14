from scipy.signal import savgol_filter
import numpy as np
import os
import scipy
import matplotlib.pyplot as plt

# add root directory to be able to import packages
# todo: make all packages installable so they can be called/imported by environment
import sys
from tqdm import tqdm, trange

#
import opexebo as op
import astropy


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
def load_locs_traces_running(fname_locs,
                             fname_traces,
                             arena_size,
                             n_frames_per_sec=20,
                             n_pixels_per_cm=15,   # not used    
                             min_vel=4):          #minimum velocity in cm/sec
    
        
        
    ####################### LOAD SPIKES ########################
    data = np.load(fname_traces,
                   allow_pickle=True)

    #
    upphases = data['F_upphase']
    #print ("traces: ", traces.shape, traces[0][:100])
    
    filtered_Fs = data['F_filtered']
    
    
    ####################### LOAD LOCATIONS ###################
    locs = np.load(fname_locs)
    print (locs.shape)
    
    #################### COMPUTE VELOCITY ####################
    
    dists = np.linalg.norm(locs[1:,:]-locs[:-1,:], axis=1)
    print (dists.shape)
    
    #
    vel_all = (dists)*(n_frames_per_sec)
    
    #
    from scipy.signal import savgol_filter

    vel_all = savgol_filter(vel_all, n_frames_per_sec, 2)

    #
    idx_stationary = np.where(vel_all<min_vel)[0]
    vel = vel_all.copy()
    vel[idx_stationary] = np.nan

    
   
    ####################### NORMALIZE SIZE OF ARENA  ###########################
    #
    min_x = np.min(locs[:,0])
    max_x = np.max(locs[:,0])

    min_y = np.min(locs[:,1])
    max_y = np.max(locs[:,1])

    #
    locs[:,0] = (locs[:,0]-min_x)/(max_x-min_x)*arena_size[0]
    locs[:,1] = (locs[:,1]-min_y)/(max_y-min_y)*arena_size[1]

    ####################### DELETE EXTRA IMAGING TIME ###########################
    rec_duration = locs.shape[0]
    
    upphases = upphases[:,:rec_duration]
    filtered_Fs = filtered_Fs[:,:rec_duration]
    

    ####################### REMOVE STATIONARY PERIODS ###########################
    
    #times = np.delete(times, idx_stationary, axis=0)
    
    locs = np.delete(locs, idx_stationary, axis=0)
    
    upphases = np.delete(upphases, idx_stationary, axis=1)
    
    filtered_Fs = np.delete(filtered_Fs, idx_stationary, axis=1)
    
    ################### COMPUTE TIMES BASED ON THE MOVING PERIODS ######################
    #
    times = np.arange(locs.shape[0])    
    
    #
    print ("Locs: ", locs.shape, " uphases: ", upphases.shape)
   
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

