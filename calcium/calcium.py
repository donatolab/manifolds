import numpy as np
import os
from tqdm import trange, tqdm
from scipy.signal import butter, lfilter, freqz, filtfilt
import matplotlib.pyplot as plt
# from tsnecuda import TSNE
import umap
from sklearn.decomposition import PCA
import pickle as pk
import scipy
import scipy.io
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import parmap
import networkx as nx
import sklearn
import pandas as pd
import cv2

#
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

#
def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

#
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

#
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def run_UMAP(data,
             n_neighbors=50,
             min_dist=0.1,
             n_components=3,
             metric='euclidean'):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )

    u = fit.fit_transform(data)

    return u

#
class Calcium():

    def __init__(self):
        self.sample_rate = 30
        #print ("Sample rate: ", self.sample_rate, "hz")

        self.verbose = False

        #
        self.keep_plot = True

        #
        self.recompute = False

        #
        self.parallel = True
        self.n_cores = 4

        # Oasis/spike parameters
        self.oasis_thresh_prefilter = 15                       # min oasis spike value that survives
        self.min_thresh_std_oasis = .1                          # upphase binarizatino step: min std for binarization of smoothed oasis curves
        self.min_width_event_oasis = 2                              # <--- min width of the window in frame times
        self.min_event_amplitude = 1                           # oasis scaled spikes: float point scaling boolean events; minimum amplitude required (removes very small amplitude events)

        # Fluorescence parameters
        self.min_thresh_std_Fluorescence_onphase = 1.5         # onphase binarization step: min x std for binarization of Fluorescence events
        self.min_thresh_std_Fluorescence_upphase = 1.5         # upphase binarization step: min x std for binarization of Fluorescence events
        self.min_width_event_Fluorescence = 15



    #
    def load_calcium(self):

        self.calcium_data = np.load(self.fname)

    #
    def load_suite2p(self):
        print ('')
        print ('')

        self.F = np.load(os.path.join(self.data_dir,
                                      'F.npy'), allow_pickle=True)
        self.Fneu = np.load(os.path.join(self.data_dir,
                                      'Fneu.npy'), allow_pickle=True)

        self.iscell = np.load(os.path.join(self.data_dir,
                                      'iscell.npy'), allow_pickle=True)

        self.ops = np.load(os.path.join(self.data_dir,
                                      'ops.npy'), allow_pickle=True)

        self.spks = np.load(os.path.join(self.data_dir,
                                      'spks.npy'), allow_pickle=True)

        self.stat = np.load(os.path.join(self.data_dir,
                                      'stat.npy'), allow_pickle=True)

        self.session_dir = os.path.join(self.data_dir,
                                   'plane0')



        ############################################################
        ################## REMOVE NON-CELLS ########################
        ############################################################
        #
        idx = np.where(self.iscell[:,0]==1)[0]
        self.F = self.F[idx]
        self.Fneu = self.Fneu[idx]

        self.spks = self.spks[idx]
        self.stat = self.stat[idx]

        #############################################################
        ########### COMPUTE GLOBAL MEAN - REMOVE MEAN ###############
        #############################################################

        self.std_global = self.compute_std_global(self.F)
        if self.verbose:
            print ("  Fluorescence data loading information")
            print ("         sample rate: ", self.sample_rate, "hz")
            print ("         self.F (fluorescence): ", self.F.shape)
            print ("         self.Fneu (neuropile): ", self.Fneu.shape)
            print ("         self.iscell (Suite2p cell classifier output): ", self.iscell.shape)
            print ("         # of good cells: ", np.where(self.iscell==1)[0].shape)
            #print ("         self.ops: ", self.ops.shape)
            print ("         self.spks (deconnvoved spikes): ", self.spks.shape)
            print ("         self.stat (footprints structure): ", self.stat.shape)
            print ("         mean std over all cells : ", self.std_global)

    def compute_std_global(self, F):
        stds = np.std(F, axis=1)
        y = np.histogram(stds, bins=np.arange(0, 100, .5))
        argmax = np.argmax(y[0])
        std_global = y[1][argmax]

        return std_global


    def load_inscopix(self):

        from numpy import genfromtxt
        data = genfromtxt(self.fname_inscopix, delimiter=',', dtype='str')

        self.F = np.float32(data[2:, 1:]).T
        self.F_times = np.float32(data[2:, 0])

        #
        self.std_global = self.compute_std_global(self.F)
        if self.verbose:
            print ("cells: ", self.F.shape)
            print ("F times: ", self.F_times.shape)
            print ("         mean std over all cells : ", self.std_global)


    #
    def standardize(self, traces):

        fname_out = os.path.join(self.root_dir,
                                 'standardized.npy')

        if True:
            #os.path.exists(fname_out)==False:
            traces_out = traces.copy()
            for k in trange(traces.shape[0], desc='standardizing'):

                temp = traces[k]
                temp -= np.median(temp)
                temp = (temp)/(np.max(temp)-np.min(temp))
        #

                #temp -= np.median(temp)
                traces_out[k] = temp



        #     np.save(fname_out, traces_out)
        # else:
        #     traces_out = np.load(fname_out)

        return traces_out

    #
    def plot_traces(self, traces, ns,
                    label='',
                    color=None,
                    alpha=1.0,
                    linewidth=1.0):

        if self.keep_plot==False:
            plt.figure()

        #
        t = np.arange(traces.shape[1])/self.sample_rate

        for ctr, k in enumerate(ns):
            if color is None:
                plt.plot(t,traces[k], label=label,
                         alpha=alpha,
                         linewidth=linewidth)
            else:
                plt.plot(t,traces[k], label=label,
                         color=color,
                         alpha=alpha,
                         linewidth=linewidth)

        # print ("np mean: ", np.mean(traces[k]))

        #plt.ylabel("First 100 neurons")
        plt.xlabel("Time (sec)")
        #plt.yticks([])
        plt.xlim(t[0],t[-1])
        #plt.show()

    #
    def plot_raster(self, ax, bn, galvo_times, track_times):

        # get front padding of raster:
        start_DC = galvo_times[0]/10000.

        # convert to frame time
        start_DC_frame_time = int(start_DC*self.sample_rate)
        print ("front padding: ", start_DC , "sec", start_DC_frame_time , ' in frame times')
        start_pad = np.zeros((bn.shape[0], start_DC_frame_time))

        # get end padding of raster
        end_DC = track_times[-1] - galvo_times[-1]/10000

        # convert to frame time
        end_DC_frame_time = int(end_DC * self.sample_rate)
        print ("end padding: ", end_DC , "sec", end_DC_frame_time , ' in frame times')
        end_pad = np.zeros((bn.shape[0], end_DC_frame_time))

        # padd the image with white space
        bn = np.hstack((start_pad, bn))
        bn = np.hstack((bn, end_pad))


        #
        ax.imshow(bn,
                   aspect='auto', cmap='Greys',
                   interpolation='none')

        # ax1.xlabel("Imaging frame")




        ax.set_ylabel("Neuron", fontsize=20)
        ax.set_xticks([])
        ax.set_ylim(0,bn.shape[0])


    def low_pass_filter(self, traces):
        #
        traces_out = traces.copy()
        for k in trange(traces.shape[0], desc='low pass filter'):
            #
            temp = traces[k]

            #
            temp = butter_lowpass_filter(temp,
                                         self.high_cutoff,
                                         self.sample_rate,
                                         order=1)
            #
            traces_out[k] = temp

        #
        return traces_out

    def load_binarization(self):

        #
        fname_out = self.fname

        if os.path.exists(fname_out) and self.recompute==False:
            data = np.load(fname_out, allow_pickle=True)
            self.F_onphase_bin = data['F_onphase']
            self.F_upphase_bin = data['F_upphase']
            self.spks = data['spks']
            self.spks_smooth_bin = data['spks_smooth_upphase']

            # raw and filtered data;
            self.F_filtered = data['F_filtered']
            self.spks_x_F = data['oasis_x_F']

            # parameters saved to file as dictionary
            self.oasis_thresh_prefilter = data['oasis_thresh_prefilter']
            self.min_thresh_std_oasis = data['min_thresh_std_oasis']
            self.min_thresh_std_Fluorescence_onphase = data['min_thresh_std_Fluorescence_onphase']
            self.min_thresh_std_Fluorescence_upphase = data['min_thresh_std_Fluorescence_upphase']
            self.min_width_event_Fluorescence = data['min_width_event_Fluorescence']
            self.min_width_event_oasis = data['min_width_event_oasis']
            self.min_event_amplitude = data['min_event_amplitude']

            print (self.F_filtered.shape)

        else:
            self.binarize_fluorescence()

    def get_footprint_contour(self, cell_id, cell_boundary='convex_hull'):
        points = np.vstack((self.stat[cell_id]['xpix'],
                            self.stat[cell_id]['ypix'])).T

        img = np.zeros((512,512),dtype=np.uint8)
        img[points[:,0],points[:,1]] = 1

        #
        if cell_boundary=='concave_hull':
            hull_points = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[0][0].squeeze()

            # check for weird single isolated pixel cells
            if hull_points.shape[0]==2:
                dists = sklearn.metrics.pairwise_distances(points)
                idx = np.where(dists==0)
                dists[idx]=1E3
                mins = np.min(dists,axis=1)

                # find pixels that are more than 1 pixel away from nearest neighbour
                idx = np.where(mins>1)[0]

                # delete isoalted points
                points = np.delete(points, idx, axis=0)

                #
                img = np.zeros((512, 512), dtype=np.uint8)
                img[points[:, 0], points[:, 1]] = 1
                hull_points = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[0][0].squeeze()

            # add last point
            hull_points = np.vstack((hull_points, hull_points[0]))


        elif cell_boundary=='convex_hull':
            #
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            hull_points = np.vstack((hull_points, hull_points[0]))

        #print ("cell: ", cell_id, "  hullpoints: ", hull_points)
        return hull_points



    def load_footprints(self):
        dims = [512, 512]

        img_all = np.zeros((dims[0], dims[1]))
        imgs = []
        contours = []
        for k in range(len(self.stat)):
            x = self.stat[k]['xpix']
            y = self.stat[k]['ypix']
            img_all[x, y] = self.stat[k]['lam']

            # save footprint
            img_temp = np.zeros((dims[0], dims[1]))
            img_temp[x, y] = self.stat[k]['lam']

            img_temp_norm = (img_temp - np.min(img_temp)) / (np.max(img_temp) - np.min(img_temp))
            imgs.append(img_temp_norm)

            contours.append(self.get_footprint_contour(k))

        imgs = np.array(imgs)

        # binarize footprints
        imgs_bin = imgs.copy() * 1E5
        imgs_bin = np.clip(imgs_bin, 0, 1)

        self.contours = contours
        self.footprints = imgs
        self.footprints_all = img_all
        self.footprints_bin = imgs_bin


    def binarize_fluorescence(self):

        fname_out = os.path.join(self.data_dir,
                                 'binarized_traces.npz'
                                 )

        if os.path.exists(fname_out)==False or self.recompute:

            ####################################################
            ########### FILTER FLUROESCENCE TRACES #############
            ####################################################
            self.high_cutoff = 1
            self.low_cutoff = 0.1

            if self.verbose:
                print ('')
                print ("  Binarization parameters: ")
                print ("        low pass filter low cuttoff: ", self.high_cutoff, "hz")
                print ("        oasis_thresh_prefilter: ", self.oasis_thresh_prefilter)
                print ("        min_thresh_std_oasis: ",  self.min_thresh_std_oasis)
                print ("        min_thresh_std_Fluorescence_onphase: ", self.min_thresh_std_Fluorescence_onphase)
                print ("        min_thresh_std_Fluorescence_upphase: ", self.min_thresh_std_Fluorescence_upphase)
                print ("        min_width_event_Fluorescence: ", self.min_width_event_Fluorescence)
                print ("        min_width_event_oasis: ", self.min_width_event_oasis)
                print ("        min_event_amplitude: ", self.min_event_amplitude)


            #
            self.F_filtered = self.low_pass_filter(self.F)

            #self.F_filtered = self.band_pass_filter(self.F)  # This has too many artifacts
            self.F_filtered -= np.median(self.F_filtered, axis=1)[None].T

            ####################################################
            ###### BINARIZE FILTERED FLUORESCENCE ONPHASE ######
            ####################################################
            stds = np.ones(self.F_filtered.shape[0])+self.std_global
            #print (" std global: ", stds, " vs std local ", np.std(self.F))

            self.F_onphase_bin = self.binarize_onphase(self.F_filtered,
                                                       stds,
                                                       self.min_width_event_Fluorescence,
                                                       self.min_thresh_std_Fluorescence_onphase)

            # detect onset of ONPHASE to ensure UPPHASES overlap at least in one location with ONPHASE
            def detect_onphase(traces):
                a = traces.copy()
                locs = []
                for k in range(a.shape[0]):
                    idx = np.where((a[k][1:]-a[k][:-1])==1)[0]
                    locs.append(idx)

                locs = np.array(locs, dtype=object)
                return locs

            onphases = detect_onphase(self.F_onphase_bin)

            ####################################################
            ###### BINARIZE FILTERED FLUORESCENCE UPPHASE ######
            ####################################################
            # THIS STEP SOMETIMES MISSES ONPHASE COMPLETELY DUE TO GRADIENT;
            # So we minimally add onphases from above
            der = np.float32(np.gradient(self.F_filtered,
                                         axis=1))
            min_slope = 0
            idx = np.where(der <= min_slope)
            F_upphase = self.F_filtered.copy()
            F_upphase[idx]=0

            #
            self.F_upphase_bin = self.binarize_onphase(F_upphase,
                                                       stds,    # use the same std array as for full Fluorescence
                                                       self.min_width_event_Fluorescence//2,
                                                       self.min_thresh_std_Fluorescence_upphase)

            # make sure that upphase data has at least on the onphase start
            # some of the cells miss this
            for k in range(len(onphases)):
                idx = np.int32(onphases[k])
                for id_ in idx:
                    self.F_upphase_bin[k,id_:id_+2] = 1

            ############################################################
            ################## THRESHOLD RAW OASIS #####################
            ############################################################
            try:
                for k in range(self.spks.shape[0]):
                    # idx = np.where(c.spks_filtered[k]<stds[k]*1.0)[0]
                    idx = np.where(self.spks[k] < self.oasis_thresh_prefilter)[0]
                    self.spks[k, idx] = 0

                ############################################################
                ################# SMOOTH OASIS SPIKES ######################
                ############################################################
                # self.spks_filtered = self.smooth_traces(self.spks, F_detrended)
                self.spks_filtered = self.smooth_traces(self.spks,
                                                        self.F_filtered)

                #################################################
                ##### SCALE OASIS BY FLUORESCENCE ###############
                #################################################
                self.spks_x_F = self.spks_filtered * self.F_filtered

                ############################################################
                ###### UPPHASE DETECTION ON OASIS SMOOTH/SCALED DATA #######
                ############################################################
                der = np.float32(np.gradient(self.spks_x_F,
                                             axis=1))
                #
                idx = np.where(der <= 0)

                #
                spks_upphase = self.spks_x_F.copy()
                spks_upphase[idx]=0

                # compute some measure of signal quality
                #sq = stds_all(spks_upphase)
                #print ('SQ: spikes :', sq)
                spks_upphase2 = spks_upphase.copy()
                spks_upphase2[idx] = np.nan
                stds = np.nanstd(spks_upphase2, axis=1)

                self.spks_upphase_bin = self.binarize_onphase(spks_upphase,
                                                              stds,
                                                              #self.min_width_event // 3,
                                                              self.min_width_event_oasis,
                                                              self.min_thresh_std_oasis)

                ############################################################
                ############# CUMULATIVE OASIS SCALING #####################
                ############################################################
                self.spks_smooth_bin = self.scale_binarized(self.spks_upphase_bin,
                                                             self.spks)
            except:
                print("   spks not available for dataset, to add it later... ")
                self.spks = np.nan
                self.spks_smooth_bin = np.nan
                self.spks_upphase_bin = np.nan
                self.oasis_x_F = np.nan
                self.spks_x_F = np.nan

            #
            if self.save_python:
                np.savez(fname_out,
                     # binarization data
                     F_onphase=self.F_onphase_bin,
                     F_upphase=self.F_upphase_bin,
                     spks=self.spks,
                     spks_smooth_upphase=self.spks_smooth_bin,

                     # raw and filtered data;
                     F_filtered = self.F_filtered,
                     oasis_x_F = self.spks_x_F,

                     # parameters saved to file as dictionary
                     oasis_thresh_prefilter=self.oasis_thresh_prefilter,
                     min_thresh_std_oasis=self.min_thresh_std_oasis,
                     min_thresh_std_Fluorescence_onphase=self.min_thresh_std_Fluorescence_onphase,
                     min_thresh_std_Fluorescence_upphase=self.min_thresh_std_Fluorescence_upphase,
                     min_width_event_Fluorescence=self.min_width_event_Fluorescence,
                     min_width_event_oasis=self.min_width_event_oasis,
                     min_event_amplitude=self.min_event_amplitude,
                     )

            #
            if self.save_matlab:
                #io.savemat("a.mat", {"array": a})

                scipy.io.savemat(fname_out.replace('npz','mat'),
                     # binarization data
                     {"F_onphase":self.F_onphase_bin,
                      "F_upphase":self.F_upphase_bin,
                      "spks":self.spks,
                      "spks_smooth_upphase":self.spks_smooth_bin,


                     # raw and filtered data;
                     "F_filtered":self.F_filtered,
                     "oasis_x_F": self.spks_x_F,

                     # parameters saved to file as dictionary
                     "oasis_thresh_prefilter":self.oasis_thresh_prefilter,
                     "min_thresh_std_oasis":self.min_thresh_std_oasis,
                     "min_thresh_std_Fluorescence_onphase":self.min_thresh_std_Fluorescence_onphase,
                     "min_thresh_std_Fluorescence_upphase":self.min_thresh_std_Fluorescence_upphase,
                     "min_width_event_Fluorescence":self.min_width_event_Fluorescence,
                     "min_width_event_oasis":self.min_width_event_oasis,
                     "min_event_amplitude":self.min_event_amplitude,
                      }
                                 )


    #
    def smooth_traces(self, traces, F_detrended):

        # params for savgol filter
        window_length = 11
        polyorder = 1
        deriv = 0
        delta = 1

        # params for exponential kernel
        M = 100
        tau = 100  # !3 sec decay
        d_exp = scipy.signal.exponential(M, 0, tau, False)
        #d_step = np.zeros(100)
        #d_step[25:75]=1

        #
        traces_out = traces.copy()

        # Smooth Oasis spikes first using savgolay filter + lowpass
        for k in trange(traces.shape[0], desc='convolving oasis with exponentional and filtering'):
            temp = traces_out[k].copy()

            # # savgol filter:
            # if False:
            #     temp = scipy.signal.savgol_filter(temp,
            #                                    window_length,
            #                                    polyorder,
            #                                    deriv = deriv,
            #                                    delta = delta)
            # convolve with an expnential function
            #else:
            temp = np.convolve(temp, d_exp, mode='full')[:temp.shape[0]]

            # if True:
            #
            #     temp =


            if True:
                temp = butter_lowpass_filter(temp, 2, 30)
            traces_out[k] = temp

        return traces_out

    #
    def band_pass_filter(self, traces):

        #print (

        #
        traces_out = traces.copy()
        for k in trange(traces.shape[0], desc='band pass filter'):
            #
            temp = traces[k]

            #
            temp = butter_bandpass_filter(temp,
                                         self.low_cutoff,
                                         self.high_cutoff,
                                         self.sample_rate,
                                         order=1)

            #
            traces_out[k] = temp

        #
        return traces_out

    def scale_binarized(self, traces, traces_scale):

        #
        from scipy.signal import chirp, find_peaks, peak_widths

        #
        for k in trange(traces.shape[0], desc='scaling binarized data'):
            temp = traces[k].copy()

            #
            peaks, _ = find_peaks(temp)  # middle of the pluse/peak

            #
            widths, heights, starts, ends = peak_widths(temp, peaks)

            xys = np.int32(np.vstack((starts, ends)).T)

            # number of time steps after peak to add oasis spikes
            buffer = 5
            for t in range(xys.shape[0]):
                # peak = np.max(F[k,xys[t,0]:xys[t,1]])
                peak = np.sum(traces_scale[k,xys[t,0]:xys[t,1]+buffer])

                temp[xys[t,0]:xys[t,1]] *= peak
                if np.max(temp[xys[t,0]:xys[t,1]])<self.min_event_amplitude:
                    #print( "Evetn too small: ", np.max(temp[xys[t,0]:xys[t,1]]),
                          # xys[t,0], xys[t,1])
                    temp[xys[t,0]:xys[t,1]+1]=0

            traces[k] = temp

        return traces

    def binarize_onphase(self,
                         traces,
                         val_scale,
                         min_width_event,
                         min_thresh_std):
        '''
           Function that converts continuous float value traces to
           zeros and ones based on some threshold

           Here threshold is set to standard deviation /10.

            Retuns: binarized traces
        '''

        #
        traces_bin = traces.copy()

        #
        for k in trange(traces.shape[0], desc='binarizing continuous traces'):
            temp = traces[k].copy()

            # find threshold crossings standard deviation based
            val = val_scale[k]
            idx1 = np.where(temp>=val*min_thresh_std)[0]  # may want to use absolute threshold here!!!

            #
            temp = temp*0
            temp[idx1] = 1

            # FIND BEGINNIGN AND ENDS OF FLUORescence above some threshold
            from scipy.signal import chirp, find_peaks, peak_widths
            peaks, _ = find_peaks(temp)  # middle of the pluse/peak
            widths, heights, starts, ends = peak_widths(temp, peaks)

            #
            xys = np.int32(np.vstack((starts, ends)).T)
            idx = np.where(widths < min_width_event)[0]
            xys = np.delete(xys, idx, axis=0)

            traces_bin[k] = traces_bin[k]*0

            # fill the data with 1s
            buffer = 0
            for p in range(xys.shape[0]):
                traces_bin[k,xys[p,0]:xys[p,1]+buffer] = 1

        return traces_bin

        #
    def binarize_derivative(self, traces, thresh=2):

        fname_out = os.path.join(self.root_dir,
                                 'binarized_derivative.npy')
        if True:
        # if os.path.exists(fname_out) == False:

            #
            traces_out = traces.copy() * 0
            traces_out_anti_aliased = traces.copy() * 0  # this generates minimum of 20 time steps for better vis

            #
            for k in trange(traces.shape[0], desc='computing derivative'):
                temp = traces[k]

                #std = np.std(temp)
                #idx = np.where(temp >= std * thresh)[0]

                #traces_out[k] = 0
                #traces_out[k, idx] = 1

                grad = np.gradient(temp)
                traces_out[k] = grad


                #
                # for id_ in idx:
                #     traces_out_anti_aliased[k, id_:id_ + 20] = 1
                #     if k > 0:
                #         traces_out_anti_aliased[k - 1, id_:id_ + 20] = 1

            np.save(fname_out, traces_out)
        else:
            traces_out = np.load(fname_out)
            traces_out_anti_aliased = traces_out.copy()

            # clip aliased data back down
            # traces_aliased = np.clip(traces_out_anti_aliased, 0,1)

        return traces_out, traces_out_anti_aliased


    #
    def binarize(self, traces, thresh = 2):

        fname_out = os.path.join(self.root_dir,
                                 'binarized.npy')
        if os.path.exists(fname_out)==False:
            traces_out = traces.copy()*0
            traces_out_anti_aliased = traces.copy()*0  # this generates minimum of 20 time steps for better vis
            for k in trange(traces.shape[0], desc='binarizing'):
                temp = traces[k]

                std = np.std(temp)

                idx = np.where(temp>=std*thresh)[0]

                traces_out[k] = 0
                traces_out[k,idx] = 1

                for id_ in idx:
                    traces_out_anti_aliased[k,id_:id_+20] = 1
                    if k>0:
                        traces_out_anti_aliased[k-1,id_:id_+20] = 1

            np.save(fname_out, traces_out)
        else:
            traces_out = np.load(fname_out)
            traces_out_anti_aliased = traces_out.copy()

            # clip aliased data back down
            # traces_aliased = np.clip(traces_out_anti_aliased, 0,1)

        return traces_out, traces_out_anti_aliased


    def compute_PCA(self, X, suffix='', recompute=True, cell_randomization=False):
        #

        # run PCA

        fname_out = os.path.join(self.root_dir, self.animal_id,
                                 self.session,
                                 'suite2p','plane0', 'pca_'+suffix+'_random_'+
                                   str(cell_randomization)+'.pkl')

        if os.path.exists(fname_out)==False or recompute:
            print ("... running PCA ...")
            pca = PCA()
            X_pca = pca.fit_transform(X)


            pk.dump(pca, open(fname_out, "wb"))

            #
            np.save(fname_out.replace('pkl','npy'), X_pca)
        else:
            with open(fname_out, 'rb') as file:
                pca = pk.load(file)

            X_pca = np.load(fname_out.replace('pkl','npy'))

        return pca, X_pca



    def compute_TSNE(self, X):
        #
        #print(self.root_dir)

        fname_out = os.path.join(self.root_dir, 'tsne.npz')
        #print ("Fname out: ", fname_out)

        try:
            data = np.load(fname_out, allow_pickle=True)
            X_tsne_gpu = data['X_tsne_gpu']

        except:

            n_components = 2
            perplexity = 100
            learning_rate = 10

            #
            X_tsne_gpu = TSNE(n_components=n_components,
                              perplexity=perplexity,
                              learning_rate=learning_rate).fit_transform(X)

            np.savez(fname_out,
                     X_tsne_gpu=X_tsne_gpu,
                     n_components=n_components,
                     perplexity=perplexity,
                     learning_rate=learning_rate
                     )


        return X_tsne_gpu


    def compute_UMAP(self, X, n_components = 3, text=''):
        #
        print(self.root_dir)

        fname_out = os.path.join(self.root_dir, text+'umap.npz')

        try:
            data = np.load(fname_out, allow_pickle=True)
            X_umap = data['X_umap']
        except:

            n_components = n_components
            min_dist = 0.1
            n_neighbors = 50
            metric = 'euclidean'

            #
            X_umap = run_UMAP(X,
                              n_neighbors,
                              min_dist,
                              n_components,
                              metric)

            np.savez(fname_out,
                     X_umap=X_umap,
                     n_components=n_components,
                     min_dist=min_dist,
                     n_neighbors=n_neighbors,
                     metric=metric
                     )

        return X_umap


    def find_sequences(self, data, thresh=1):

        #
        segs = []
        seg = []
        seg.append(0)

        clrs = []
        ctr = 0
        clrs.append(ctr)

        #
        for k in trange(1, data.shape[0], 1, desc='finding sequences'):

            temp = dist = np.linalg.norm(data[k] - data[k - 1])

            if temp <= thresh:
                seg.append(k)
                clrs.append(ctr)
            else:
                segs.append(seg)
                seg = []
                seg.append(k)

                #
                ctr = np.random.randint(10000)
                clrs.append(ctr)

        # add last segment if missed:
        if len(segs[-1]) > 1:
            segs.append(seg)

        return segs, clrs

    def find_candidate_neurons_overlaps(self):

        dist_corr_matrix = []
        for index, row in self.df_overlaps.iterrows():
            cell1 = int(row['cell1'])
            cell2 = int(row['cell2'])
            percent1 = row['percent_cell1']
            percent2 = row['percent_cell2']

            if cell1 < cell2:
                corr = self.corr_array[cell1, cell2, 0]
            else:
                corr = self.corr_array[cell2, cell1, 0]

            dist_corr_matrix.append([cell1, cell2, corr, max(percent1, percent2)])

        dist_corr_matrix = np.vstack(dist_corr_matrix)

        #####################################################
        idx1 = np.where(dist_corr_matrix[:, 3] >= self.corr_max_percent_overlap)[0]
        idx2 = np.where(dist_corr_matrix[idx1, 2] >= self.corr_threshold)[0]
        idx3 = idx1[idx2]

        #
        self.candidate_neurons = dist_corr_matrix[idx3][:, :2]

        return self.candidate_neurons



    def find_candidate_neurons_centers(self):
        dist_corr_matrix = []

        for k in trange(self.dists.shape[0], desc='finding candidate neurons'):
            for p in range(k + 1, self.dists.shape[0]):
                dist = self.dists[k, p]
                corr = self.corr_array[k, p, 0]
                dist_corr_matrix.append([dist, corr, k, p])

        dist_corr_matrix = np.vstack(dist_corr_matrix)

        # ####################################################
        # ####### GET NEURONS WITH SUSPICIOUS PROPERTIES #####
        # ####################################################
        idx1 = np.where(dist_corr_matrix[:, 0] <= self.corr_min_distance)[0]
        idx2 = np.where(dist_corr_matrix[idx1, 1] >= self.corr_threshold)[0]

        #
        idx3 = idx1[idx2]

        self.candidate_neurons = dist_corr_matrix[idx3][:, 2:]

        return self.candidate_neurons

    def make_correlated_neuron_graph(self):
        adjacency = np.zeros((self.F.shape[0],
                              self.F.shape[0]))
        for i in self.candidate_neurons:
            adjacency[int(i[0]), int(i[1])] = 1

        G = nx.Graph(adjacency)
        G.remove_nodes_from(list(nx.isolates(G)))

        self.G = G



    def delete_duplicate_cells(self):
        from time import sleep

        a = nx.connected_components(self.G)
        removed_cells = []
        tot, a = it_count(a)
        with tqdm(total=tot) as pbar:
            ctr=0
            for nn in a:
                if self.corr_delete_method=='lowest_snr':
                    good_ids, removed_ids = del_lowest_snr(nn, self)
                elif self.corr_delete_method=='highest_connected':
                    good_ids, removed_ids = del_highest_connected_nodes(nn, self)
                # print (removed_ids)
                removed_cells.append(removed_ids)
                #sleep(0.01)
                pbar.update(ctr)
                ctr+=1
        #
        if len(removed_cells)>0:
            removed_cells = np.hstack(removed_cells)
        else:
            removed_cells = []
        clean_cells = np.delete(np.arange(self.F.shape[0]),
                              removed_cells)
        self.clean_cell_ids = clean_cells

        return self.clean_cell_ids

    def plot_corr_vs_distance(self):
        dist_corr_matrix = []
        for k in range(self.dists.shape[0]):
            for p in range(k + 1, self.dists.shape[0]):
                if k in self.clean_cell_ids and p in self.clean_cell_ids:
                    dist = self.dists[k, p]
                    corr = self.corr_array[k, p, 0]
                    dist_corr_matrix.append([dist, corr, k, p])

        dist_corr_matrix = np.vstack(dist_corr_matrix)
        plt.scatter(dist_corr_matrix[:, 0], dist_corr_matrix[:, 1],
                    alpha=.3,
                    edgecolor='black')

        plt.ylabel("correlation")
        plt.xlabel("distance between centres (pixels)")

        # ####################################################
        # ####### GET NEURONS WITH SUSPICIOUS PROPERTIES #####
        # ####################################################
        if self.show_outliers:
            idx1 = np.where(dist_corr_matrix[:, 0] <= self.corr_min_distance)[0]
            idx2 = np.where(dist_corr_matrix[idx1, 1] >= self.corr_threshold)[0]

            #
            idx3 = idx1[idx2]
            plt.scatter(dist_corr_matrix[idx3, 0],
                        dist_corr_matrix[idx3, 1],
                        alpha=.1,
                        edgecolor='red')


    def remove_duplicate_neurons(self):

        # get pairwise distances between neuron centres; distsupper is zeroed out array

        #if self.deduplication_method == 'centre_distance':
        self.dists, self.dists_upper = find_inter_cell_distance(self.footprints)
        #elif self.deduplication_method == 'overlap':
        self.df_overlaps = generate_cell_overlaps(self)

        # compute correlations between neurons
        rasters = self.F_filtered   # use fluorescence filtered traces
        self.corrs = compute_correlations(rasters, self)
        self.corr_array = make_correlation_array(self.corrs, rasters)

        # find neurnos that are below threhsolds
        if self.deduplication_method =='centre_distance':
            self.candidate_neurons = self.find_candidate_neurons_centers()
        elif self.deduplication_method == 'overlap':
            self.candidate_neurons = self.find_candidate_neurons_overlaps()


        # find connected neurons and remove singles
        self.make_correlated_neuron_graph()

        #
        self.delete_duplicate_cells()

def parallel_network_delete(nn, ):
    pass


def it_count(it):
    import itertools
    tmp_it, new_it = itertools.tee(it)
    return sum(1 for _ in tmp_it), new_it


def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)


#
def get_correlations(ids, c):
    corrs = []
    for i in range(ids.shape[0]):
        for ii in range(i + 1, ids.shape[0], 1):
            if ids[i] < ids[ii]:
                corrs.append(c.corr_array[ids[i], ids[ii], 0])
            else:
                corrs.append(c.corr_array[ids[ii], ids[i], 0])

    corrs = np.array(corrs)

    return corrs


def del_highest_connected_nodes(nn, c):
    # get correlations for all cells in group
    ids = np.array(list(nn))
    corrs = get_correlations(ids, c)
    # print("ids: ", ids, " starting corrs: ", corrs)

    # find lowest SNR neuron
    removed_cells = []
    while np.max(corrs) > c.corr_threshold:

        n_connections = []
        snrs = []
        for n in ids:
            temp1 = signaltonoise(c.F_filtered[n])
            snrs.append(temp1)
            temp2 = c.G.edges([n])
            n_connections.append(len(temp2))

        # find max # of edges
        max_edges = np.max(n_connections)
        idx = np.where(n_connections == max_edges)[0]

        # if a single max exists:
        if idx.shape[0] == 1:
            idx2 = np.argmax(n_connections)
            removed_cells.append(ids[idx2])
            ids = np.delete(ids, idx2, 0)
        # else select the lowest SNR among the nodes
        else:
            snrs = np.array(snrs)
            snrs_idx = snrs[idx]
            idx3 = np.argmin(snrs_idx)

            if c.verbose:
                print("multiple matches found: ", snrs, snrs_idx, idx3)
            removed_cells.append(ids[idx[idx3]])
            ids = np.delete(ids, idx[idx3], 0)

        if ids.shape[0] == 1:
            break

        corrs = get_correlations(ids, c)
        if c.verbose:
            print("ids: ", ids, "  corrs: ", corrs)

    good_cells = ids
    return good_cells, removed_cells


#
def del_lowest_snr(nn, c):

    '''
        input
        nn
        c.corr_threshold
        c.F_filtered


    :param nn:
    :param c:
    :return:
    '''
    # get correlations for all cells in group
    ids = np.array(list(nn))
    corrs = get_correlations(ids, c)
    # print("ids: ", ids, " starting corrs: ", corrs)

    # find lowest SNR neuron
    removed_cells = []
    while np.max(corrs) > c.corr_threshold:
        snrs = []
        for n in ids:
            temp = signaltonoise(c.F_filtered[n])
            snrs.append(temp)

        # print ("ids: ", ids, "  snrs: ", snrs)
        idx = np.argmin(snrs)
        removed_cells.append(ids[idx])
        ids = np.delete(ids, idx, 0)

        if ids.shape[0] == 1:
            break

        corrs = get_correlations(ids, c)
    # print ("ids: ", ids, "  corrs: ", corrs)

    good_cells = ids
    return good_cells, removed_cells



def array_row_intersection(a, b):
    tmp = np.prod(np.swapaxes(a[:, :, None], 1, 2) == b, axis=2)
    return a[np.sum(np.cumsum(tmp, axis=0) * tmp == 1, axis=1).astype(bool)]


def find_overlaps1(ids, footprints):
    #
    intersections = []
    for k in ids:
        temp1 = footprints[k]
        idx1 = np.vstack(np.where(temp1 > 0)).T

        #
        for p in range(k + 1, footprints.shape[0], 1):
            temp2 = footprints[p]
            idx2 = np.vstack(np.where(temp2 > 0)).T
            res = array_row_intersection(idx1, idx2)

            #
            if len(res) > 0:
                percent1 = res.shape[0] / idx1.shape[0]
                percent2 = res.shape[0] / idx2.shape[0]
                intersections.append([k, p, res.shape[0], percent1, percent2])
    #
    return intersections

#
def find_overlaps(ids, footprints):
    #
    intersections = []
    for k in ids:
        temp = footprints[k]
        idx1 = np.vstack(np.where(temp > 0)).T

        #
        for p in range(k + 1, footprints.shape[0], 1):
            temp = footprints[p]
            idx2 = np.vstack(np.where(temp > 0)).T

            res = array_row_intersection(idx1, idx2)
            if len(res) > 0:
                intersections.append([k, p, res])

    return intersections


#
def make_overlap_database(res):
    data = []
    for k in range(len(res)):
        for p in range(len(res[k])):
            # print (res[k][p])
            data.append(res[k][p])

    df = pd.DataFrame(data, columns=['cell1', 'cell2',
                                     'pixels_overlap',
                                     'percent_cell1',
                                     'percent_cell2'])

    return (df)


#
def find_inter_cell_distance(footprints):
    locations = []
    for k in trange(footprints.shape[0], desc='finding medians'):
        temp = footprints[k]
        centre = np.median(np.vstack(np.where(temp > 0)).T, axis=0)
        locations.append(centre)

    locations = np.vstack(locations)
    dists = sklearn.metrics.pairwise.euclidean_distances(locations)

    # zero out bottom part of matrix for redundancy
    dists_upper = np.triu(dists, -1)

    idx = np.where(dists == 0)
    dists[idx] = np.nan

    return dists, dists_upper


#
def compute_correlations(rasters, c):
    fname_out = os.path.join(c.root_dir,
                             c.animal_id,
                             c.session,
                             'suite2p',
                             'plane0',
                             'cell_correlations.npy'
                             )
    if os.path.exists(fname_out) == False:
        #
        corrs = []
        for k in trange(rasters.shape[0],desc='computing intercell correlation'):
            temp1 = rasters[k]
            #
            for p in range(k + 1, rasters.shape[0], 1):
                temp2 = rasters[p]
                corr = scipy.stats.pearsonr(temp1,
                                            temp2)

                corrs.append([k, p, corr[0], corr[1]])

        corrs = np.vstack(corrs)

        np.save(fname_out, corrs)
    else:
        corrs = np.load(fname_out)

    return corrs


def make_correlation_array(corrs, rasters):
    # data = []
    corr_array = np.zeros((rasters.shape[0], rasters.shape[0], 2), 'float32')

    for k in trange(len(corrs),desc='getting correlations and pvals'):
        cell1 = int(corrs[k][0])
        cell2 = int(corrs[k][1])
        pcor = corrs[k][2]
        pval = corrs[k][3]

        corr_array[cell1, cell2, 0] = pcor
        corr_array[cell1, cell2, 1] = pval

    return corr_array


def generate_cell_overlaps(c):
    fname_out = os.path.join(c.root_dir,
                             c.animal_id,
                             c.session,
                             'suite2p',
                             'plane0',
                             'cell_overlaps.pkl'
                             )

    if os.path.exists(fname_out) == False:

        ids = np.array_split(np.arange(c.footprints.shape[0]), 30)

        if c.parallel:
            res = parmap.map(find_overlaps1,
                         ids,
                         c.footprints,
                         pm_processes=c.n_cores,
                         pm_pbar=True)
        else:
            res = []
            for k in trange(len(ids)):
                res.append(find_overlaps1(ids[k],
                                          c.footprints))

        df = make_overlap_database(res)

        df.to_pickle(fname_out)  # where to save it, usually as a .pkl

    else:
        df = pd.read_pickle(fname_out)

    return df

def alpha_shape(points, alpha=0.6):

    from shapely.ops import cascaded_union, polygonize
    from scipy.spatial import Delaunay
    import shapely.geometry as geometry
    """
    Compute the alpha shape (concave hull) of a set
    of points.
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull

    #coords = np.array([point.coords[0] for point in points])
    coords = points

    #
    tri = Delaunay(coords)
    triangles = coords[tri.vertices]
    a = ((triangles[:,0,0] - triangles[:,1,0]) ** 2 + (triangles[:,0,1] - triangles[:,1,1]) ** 2) ** 0.5
    b = ((triangles[:,1,0] - triangles[:,2,0]) ** 2 + (triangles[:,1,1] - triangles[:,2,1]) ** 2) ** 0.5
    c = ((triangles[:,2,0] - triangles[:,0,0]) ** 2 + (triangles[:,2,1] - triangles[:,0,1]) ** 2) ** 0.5
    s = ( a + b + c ) / 2.0
    areas = (s*(s-a)*(s-b)*(s-c)) ** 0.5
    circums = a * b * c / (4.0 * areas)
    filtered = triangles[circums < (1.0 / alpha)]
    edge1 = filtered[:,(0,1)]
    edge2 = filtered[:,(1,2)]
    edge3 = filtered[:,(2,0)]
    edge_points = np.unique(np.concatenate((edge1,edge2,edge3)), axis = 0).tolist()
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))

    return cascaded_union(triangles), edge_points