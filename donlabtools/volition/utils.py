# Visualisation
import matplotlib.pyplot as plt

# Computing
import numpy as np
from scipy import io
from scipy import stats
import sys
import pickle
from tqdm import tqdm

# Files
import pandas as pd
import os
import glob
from tqdm import trange


import statsmodels.api as sm
#import statsmodels as sm
import math
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from scipy.stats import norm
from scipy.spatial.distance import cdist
import numpy as np

def preprocess_data2D(root_dir, session, fname_neural, fname_locs):
    raw_data = np.array(pd.read_csv(os.path.join(root_dir,session, fname_neural)))
    raw_data =  raw_data[:,1:-25]
    data2 = raw_data.transpose()
    raw_data = np.array(data2)

    neural_data = raw_data[:,1:-25]

    time = np.arange(0,36000)

    #locs_cm, locs_dec = load_data(path, filename)
    locs_cm = np.load(os.path.join(root_dir,session,fname_locs))
    locs = locs_cm

    # partition space 
    partition_size = 10               # partition size in centimeters
    box_width, box_length = 80, 80    # length in centimeters

    # centre to 0,0
    locs[:,0] = locs[:,0]-np.min(locs[:,0])
    locs[:,1] = locs[:,1]-np.min(locs[:,1])

    xmax = np.max(locs[:,0])
    ymax = np.max(locs[:,1])
    print (xmax,ymax)
    pixels_per_cm = xmax / box_width

    # convert DLC coords to cm
    locs_cm = locs_cm/pixels_per_cm

    print("neural_data: ", neural_data.shape)
    print ("locs_cm: ", locs_cm.shape)
    return neural_data, locs_cm, time


def load_tracks_2D(fname_locs,
                     fname_bin):


    #
    d = np.load(fname_bin,
                  allow_pickle=True)
    F_upphase = d["F_upphase"]
    #print ("bin upphase: ", F_upphase.shape)

    # start of recording as some imaging didn't start same time as DLC tracking

    #
    locs = np.load(fname_locs)

    #print ("All DLC locations: ", locs.shape)
    locs = locs[0:36000]
    #print ("Final locations shape: ", locs.shape)
    F_upphase = F_upphase[0:36000]
    #print ("bin upphase: ", F_upphase.shape)

    return (F_upphase, locs)

def get_data_decoders_2D(fname_locs,
                         fname_bin,
                         partition_size=5):

    F_upphase, locs = load_tracks_2D(fname_locs,
                                           fname_bin)
    #print ("locs: ", locs.shape)
    # partition space into specific parcels
    F_upphase = F_upphase.T
    F_upphase = F_upphase[0:36000]
    #print ("F: ", F_upphase.shape)

    # bin data
    bin_size = 7
    run_binnflag = False
    if run_binnflag:
        sum_flag = True
        f_binned= run_binning(F_upphase, bin_size, sum_flag)
    else:
        f_binned = F_upphase
    #print (f_binned.shape)

    # 
    if run_binnflag:
        sum_flag = False
        locs_binned = run_binning(locs, bin_size, sum_flag)
    else:
        locs_binned = locs
    #print ("locs_binned: ", locs_binned.shape)


    # partition space 
    box_width, box_length = 80, 80    # length in centimeters

    # centre to 0,0
    locs[:,0] = locs[:,0]-np.min(locs[:,0])
    locs[:,1] = locs[:,1]-np.min(locs[:,1])

    xmax = np.max(locs[:,0])
    ymax = np.max(locs[:,1])
    #print (xmax,ymax)
    pixels_per_cm = xmax / box_width

    # convert DLC coords to cm
    locs_cm = locs_binned/pixels_per_cm
    #print ("locs_cm: ", locs_cm.shape)

    # partiiont the square box;  # it will be a square array
    # it just holds the numbers 1..64 for rexample, for an 8 x 8 partition
    partition = np.zeros((box_width//partition_size, 
                          box_width//partition_size),
                          )

    # it contains matching of pop vectors to a specific bin (ranging from 1..64)
    locs_partitioned = np.zeros(locs_cm.shape[0])

    # list of 64 lists containing the population vector identities for each bin location
    partition_times = []

    ctrx = 0
    bin_ctr = 0
    for x in range(0,box_width,partition_size):
        idx = np.where(np.logical_and(locs_cm[:,0]>=x,
                                      locs_cm[:,0]<x+partition_size))[0]
        ctry = 0
        for y in range(0,box_width,partition_size):
            idy = np.where(np.logical_and(locs_cm[:,1]>=y,
                                          locs_cm[:,1]<y+partition_size))[0]

            # find intersection of idx and idy as the times when the mouse was in 
            #   partition bin: ctrx, ctry
            idx3 = np.intersect1d(idx, idy)
            #print (idx3)
            locs_partitioned[idx3]=bin_ctr

            partition_times.append(idx3)    

            #
            partition[ctry,ctrx]=bin_ctr

            #
            ctry+=1
            bin_ctr+=1

        ctrx+=1
    
    return f_binned, locs_partitioned, partition_times, partition, box_width, box_length, locs_cm,partition_size


def get_neural_data_and_locs(root_dir,
                             animal_id,
                             session_id):
    
    #
    from utils import get_data_decoders_2D

    fname_csv = os.path.join(root_dir,
                             animal_id,
                             session_id)
    fname_csv = glob.glob(fname_csv + '/*0000.csv')[0]

    fname_locs = fname_csv[:-4] + '_locs.npy'

    #
    temp = os.path.join(root_dir,
                        animal_id,
                        session_id,
                        '*binarized_traces*.npz')
    # print ("temp: ", temp)
    fname_bin = glob.glob(temp)[0]

    #
    # print ("TODO: use start offset for mouse 7050: ")
    fname_start = os.path.join(root_dir,
                               animal_id,
                               session_id,
                               'start.txt')

    # Get the data in bins
    partition_size = 10

    (neural_data,
     locs_partitioned,
     partition_times,
     partition,
     box_width,
     box_length,
     locs_cm,
     partition_size,
     ) = get_data_decoders_2D(fname_locs,
                              fname_bin,
                              partition_size=partition_size)

    return neural_data, locs_cm


def compute_speed(locs_cm, time_frame = 0.05, smooth = True, frames_smooth = 30):
    if smooth:
        kernel = np.ones(frames_smooth) / frames_smooth
        locs_cm[:,0] = np.convolve(locs_cm[:,0], kernel, mode='same')
        locs_cm[:,1] = np.convolve(locs_cm[:,1], kernel, mode='same')

    speed = np.sqrt((np.diff(locs_cm[:,0]))**2+(np.diff(locs_cm[:,1]))**2)/(time_frame)
    speed = np.append(speed, 0)
    if smooth:
        kernel = np.ones(frames_smooth) / frames_smooth
        speed = np.convolve(speed, kernel, mode='same')
    return speed


def is_mobile_smooth(speed,
                     threshold=5,
                     window=20,
                     order = 4):
    from scipy.signal import savgol_filter
    speed = savgol_filter(speed, window, order)  # window size 13, polynomial order 5

    mobile = speed > threshold
    mobile = mobile.astype(int)
    idx_mob = np.where(mobile == 1)[0]
    idx_imm = np.where(mobile == 0)[0]
    return mobile, idx_mob, idx_imm



def is_mobile(speed, threshold=5):
    mobile = speed>threshold
    mobile = mobile.astype(int)
    idx_mob = np.where(mobile == 1)[0]
    idx_imm = np.where(mobile == 0)[0]
    return mobile, idx_mob, idx_imm

def plot_results(y, y_pred, name=''):
    x = np.arange(len(y[:,0]))
    frame_rate = 1200
    time = x/frame_rate
    fig = plt.figure(figsize=(8,2))
    plt.plot(time, y[:,0], label = 'real', alpha=0.8)
    plt.plot(time, y_pred[:,0], label = 'predicted', alpha=0.8)
    plt.title(name)
    plt.xlim(0,30)
    plt.ylim(0,80)
    plt.xlabel('time (min)')
    plt.ylabel('x position (cm)')
    plt.legend(loc='upper center', bbox_to_anchor=((0.7, -0.15)))
    fig.savefig(name+"_xpos.png", dpi=600, bbox_inches="tight")
    
    fig = plt.figure(figsize=(8,2))
    plt.plot(time, y[:,1], label = 'real', alpha=0.8)
    plt.plot(time, y_pred[:,1], label = 'predicted', alpha=0.8)
    plt.xlim(0,30)
    plt.ylim(0,80)
    plt.title(name)
    plt.xlabel('time(min)')
    plt.ylabel('y position (cm)')
    plt.legend(loc='upper center', bbox_to_anchor=((0.7, -0.15)))
    fig.savefig(name+"_ypos.png", dpi=600, bbox_inches="tight")
    
def get_distance(y, y_pred):
    x_dec = y_pred[:,0]
    y_dec = y_pred[:,1]
    x_cm = y[:,0]
    y_cm = y[:,1]
    distance = np.sqrt((x_dec-x_cm)**2+(y_dec-y_cm)**2)
    return distance

def random_distance(y, y_pred):
    y_shuffle = y.copy()
    np.random.shuffle(y_shuffle)
    print (y_shuffle.shape)
    print(y.shape)
    shuffled_distance = get_distance(y_shuffle, y_pred)
    return shuffled_distance

def plot_histogram_distances(y, y_pred, name=''):
    distance = get_distance(y, y_pred)
    print(distance)
    shuffled_distance = random_distance(y, y_pred)

    fig = plt.figure(figsize=(2,2))
    plt.hist(distance, label = 'real', alpha = 0.5)
    plt.hist(shuffled_distance, label = 'random', color = 'green', alpha = 0.5)
    plt.title(name)
    plt.xlabel('Distance error (cm)')
    plt.ylabel('Frequency')
    plt.legend(loc='best')
    fig.savefig(name+"_hist.png", dpi=600, bbox_inches="tight")
    return

def filter_times(partition_times, idx):
    mob_times = []
    for i in range(len(partition_times)):
        indices=np.argwhere(np.isin(partition_times[i], idx))[:,0]
        mob_times.append(partition_times[i][indices])
    return mob_times

def neural_loc_to_y(neural_location, box_width, partition_size):
    x = neural_location//(box_width//partition_size)+0.5
    y = neural_location%(box_width//partition_size)+0.5
    x = x*partition_size
    y = y*partition_size
    y_projection = np.vstack((x,y)).T
    return y_projection

# Get the location of the mouse in case it does not exists
def load_csv(fname_csv):

    from numpy import genfromtxt
    locs = genfromtxt(fname_csv, 
                      delimiter=',', 
                      dtype='str')

    #print ("Note... saving all locations regardless of likelihood...")
    #print (locs.shape)

    return locs


from sklearn.decomposition import PCA

class ProjectionDecoder(object):

    """
    Class for the Nearest Neighbor Decoder
    Parameters
    ----------
    res:int, default=100
        resolution of predicted values
        This is the number of bins to divide the outputs into (gget_data_decoders_2Doing from minimum to maximum)
        larger values will make decoding slower
    """

    def __init__(self,res=10):
        self.res=res
        return
    
    def fit_pca(self, X_train):
        self.pca = PCA(svd_solver='full')
        self.pca.fit(X_train)
        return
    
    def transform_pca(self, X):
        X_pca = self.pca.transform(X)
        return X_pca

    def fit(self,X_train ,y_train, dimensionality):

        """
        Train Naive Bayes Decoder
        Parameters
        ----------
        X_b_train: numpy 2d array of shape [n_samples,n_neurons]
            This is the neural training data.
            See example file for an example of how to format the neural data correctly
        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted (training data)
        """
        self.fit_pca(X_train)
        X_train_pca = self.transform_pca(X_train)
        X_train_pca = X_train_pca[:,:dimensionality]
    
        # compute average neural state at each spatial location
        # this can be done on all data or PCA version up to some dimensionality
        ave_states = []
        loc_idx = []
        for i in range(int(max(y_train)+1)):
            idx = np.where(y_train == i)
            loc_idx.append(idx)

        for k in range(len(loc_idx)):
            temp = X_train_pca[loc_idx[k]]
            #print ("temp: ", temp.shape)
            ave = np.nanmedian(temp,axis=0)
            ave_states.append(ave)

        data_ave_states = np.vstack(ave_states)
        self.dimensionality = dimensionality
        self.ave = data_ave_states
        return data_ave_states
    
    def predict(self, X):

        """
        Train Naive Bayes Decoder
        Parameters
        ----------
        X_b_train: numpy 2d array of shape [n_samples,n_neurons]
            This is the neural training data.
            See example file for an example of how to format the neural data correctly
        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted (training data)
        """
        X_pca = self.transform_pca(X)
    
        y_pred = np.zeros(X_pca.shape[0],dtype='int32')
        #print ("data tiraged average: ", data_triaged_ave)
        for k in range(X_pca.shape[0]-1):

            # find location of current state in PCA-space
            temp_state = X_pca[k,:self.dimensionality]  # grab only first n comps
            diff = np.abs(self.ave-temp_state).sum(axis=1)
            idx1 = np.nanargmin(diff)
            y_pred[k] = idx1

        return y_pred
    
    
def glm_run(Xr, Yr, X_range):

    #
    # idx = np.where(Yr!=0)[0]
    # print (Xr.shape, Yr.shape, idx.shape)
    # # Xr = Xr[idx]
    # # Yr = Yr[idx]
    # # print (idx.shape, Xr.shape, Yr.shape)
    # #
    # # print (Xr[:50])


    X2 = sm.add_constant(Xr)
    #X2 = sm.tools.tools.add_constant(Xr)
    poiss_model = sm.GLM(Yr, X2, family=sm.families.Poisson())
    try:
        glm_results = poiss_model.fit()
        Y_range=glm_results.predict(sm.add_constant(X_range))
    except np.linalg.LinAlgError:
        print("\nWARNING: LinAlgError")
        Y_range=np.mean(Yr)*np.ones([X_range.shape[0],1])

    return Y_range


class NaiveBayesRegression(object):

    """
    Class for the Naive Bayes Decoder
    Parameters
    ----------
    res:int, default=100
        resolution of predicted values
        This is the number of bins to divide the outputs into (going from minimum to maximum)
        larger values will make decoding slower
    """

    def __init__(self,res=10):
        self.res=res
        return

    def fit(self,X_b_train,y_train):

        """
        Train Naive Bayes Decoder
        Parameters
        ----------
        X_b_train: numpy 2d array of shape [n_samples,n_neurons]
            This is the neural training data.
            See example file for an example of how to format the neural data correctly
        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted (training data)
        """

        #### FIT TUNING CURVE ####
        #First, get the output values (x/y position or velocity) that we will be creating tuning curves over
        #Create the range for x and y (position/velocity) values
        input_x_range=np.arange(np.min(y_train[:,0]),np.max(y_train[:,0])+.01,np.round((np.max(y_train[:,0])-np.min(y_train[:,0]))/self.res))
        input_y_range=np.arange(np.min(y_train[:,1]),np.max(y_train[:,1])+.01,np.round((np.max(y_train[:,1])-np.min(y_train[:,1]))/self.res))
        #Get all combinations of x/y values
        input_mat=np.meshgrid(input_x_range,input_y_range)
        #Format so that all combinations of x/y values are in 2 columns (first column x, second column y). This is called "input_xy"
        xs=np.reshape(input_mat[0],[input_x_range.shape[0]*input_y_range.shape[0],1])
        ys=np.reshape(input_mat[1],[input_x_range.shape[0]*input_y_range.shape[0],1])
        input_xy=np.concatenate((xs,ys),axis=1)


        #Create tuning curves
        num_nrns=X_b_train.shape[1] #Number of neurons to fit tuning curves for
        tuning_all=np.zeros([num_nrns,input_xy.shape[0]]) #Matrix that stores tuning curves for all neurons

        #Loop through neurons and fit tuning curves
        for j in trange(num_nrns): #Neuron number
            X_in = X_b_train[:,j]
            try:
                tuning=glm_run(y_train,
                           X_in,
                           #X_b_train[:,j:j+1],
                           input_xy)

                #Enter tuning curves into matrix
                tuning_all[j,:]=np.squeeze(tuning)
            except:
                print ("ERROR in CELL")


        #Save tuning curves to be used in "predict" function
        self.tuning_all=tuning_all
        self.input_xy=input_xy

        #Get information about the probability of being in one state (position/velocity) based on the previous state
        #Here we're calculating the standard deviation of the change in state (velocity/acceleration) in the training set
        n=y_train.shape[0]
        dx=np.zeros([n-1,1])
        for i in range(n-1):
            dx[i]=np.sqrt((y_train[i+1,0]-y_train[i,0])**2+(y_train[i+1,1]-y_train[i,1])**2) #Change in state across time steps
        std=np.sqrt(np.mean(dx**2)) #dx is only positive. this gets approximate stdev of distribution (if it was positive and negative)
        self.std=std #Save for use in "predict" function

        #Get probability of being in each state - we are not using this since it did not help decoding performance
        # n_x=np.empty([input_xy.shape[0]])
        # for i in range(n):
        #     loc_idx=np.argmin(cdist(y_train[0:1,:],input_xy))
        #     n_x[loc_idx]=n_x[loc_idx]+1
        # p_x=n_x/n
        # self.p_x=p_x

    def predict(self,X_b_test,y_test):

        """
        Predict outcomes using trained tuning curves
        Parameters
        ----------
        X_b_test: numpy 2d array of shape [n_samples,n_features]
            This is the neural data being used to predict outputs.
        y_test: numpy 2d array of shape [n_samples,n_outputs]
            The actual outputs
            This parameter is necesary for the NaiveBayesDecoder  (unlike most other decoders)
            because the first value is nececessary for initialization
        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        #Get values saved in "fit" function
        tuning_all=self.tuning_all
        input_xy=self.input_xy
        std=self.std

        #Get probability of going from one state to the next
        dists = squareform(pdist(input_xy, 'euclidean')) #Distance between all states in "input_xy"
        #Probability of going from one state to the next, based on the above calculated distances
        #The probability is calculated based on the distances coming from a Gaussian with standard deviation of std
        prob_dists=norm.pdf(dists,0,std)

        #Setting the same probability for all the places
        prob_dists =np.ones(prob_dists.shape)

        #Initializations; THIS IS CHACKY
        # TODO: implement this correctly
        loc_idx= np.argmin(cdist(y_test[0:1,:],input_xy)) #The index of the first location
        num_nrns=tuning_all.shape[0] #Number of neurons

        #
        y_test_predicted=np.empty([X_b_test.shape[0],2]) #Initialize matrix of predicted outputs
        num_ts=X_b_test.shape[0] #Number of time steps we are predicting

        #Loop across time and decode
        for t in trange(num_ts):
            rs=X_b_test[t,:] #Number of spikes at this time point (in the interval we've specified including bins_before and bins_after)

            probs_total=np.ones([tuning_all[0,:].shape[0]]) #Vector that stores the probabilities of being in any state based on the neural activity (does not include probabilities of going from one state to the next)
            for j in range(num_nrns): #Loop across neurons
                lam=np.copy(tuning_all[j,:]) #Expected spike counts given the tuning curve
                r=rs[j] #Actual spike count
                probs=np.exp(-lam)*lam**r/math.factorial(r) #Probability of the given neuron's spike count given tuning curve (assuming poisson distribution)
                probs_total=np.copy(probs_total*probs) #Update the probability across neurons (probabilities are multiplied across neurons due to the independence assumption)
            prob_dists_vec=np.copy(prob_dists[loc_idx,:]) #Probability of going to all states from the previous state
            probs_final=probs_total*prob_dists_vec #Get final probability (multiply probabilities based on spike count and previous state)
            # probs_final=probs_total*prob_dists_vec*self.p_x #Get final probability when including p(x), i.e. prior about being in states, which we're not using
            loc_idx=np.argmax(probs_final) #Get the index of the current state (that w/ the highest probability)
            y_test_predicted[t,:]=input_xy[loc_idx,:] #The current predicted output

        return y_test_predicted #Return predictions


def get_cmap_2d():
    xlist = []
    ylist = []
    colorlist = []

    for i in range(0, 8):
        for j in range(0, 8):
            xlist.append(i)
            ylist.append(j)
            if i > j:
                colorlist.append(((float(32 * i / 255), float((255 - 32 * i) / 255), float(32 * j / 255))))
            else:
                colorlist.append(((float(32 * i / 255), float((255 - 32 * j) / 255), float(32 * j / 255))))

    if False:
        fig = plt.figure(figsize=(4, 4))
        plt.scatter(xlist, ylist, c=colorlist, edgecolor='none', s=1000, marker='s')
        plt.show()

    cmap = plt.get_cmap('autumn_r')
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', colorlist[0:], cmap.N)

    return cmap


#
def get_active_frames(neural_data, min_activity = 1):

    #print ("min spikes: ", min_spikes)
    # Remove time bins with no neural activity
    good_times=np.where(neural_data.sum(axis=1) >= min_activity)[0]

    return good_times

#
def remove_zero_activity_frames(neural_data, labels):

    #print ("min spikes: ", min_spikes)
    # Remove time bins with no neural activity
    good_times=np.where(neural_data.sum(axis=1) > 0)[0]

    return neural_data[good_times], labels[good_times]
#
def find_active_times_and_cells(neural_data, min_spikes):

    X=neural_data

    #print ("min spikes: ", min_spikes)
    # Remove time bins with no neural activity
    good_times=np.where(X.sum(axis=1) > 0)[0]
    print ("Frames with non-zero activity: ", good_times.shape[0]/20., " sec.")

    # Find non-active neurons
    good_cells = np.where(X.sum(axis=0)>=min_spikes)[0]
    print ("# cells with some activity in the period: ", good_cells.shape)

    return good_times, good_cells

    
def preprocess_bayesian2(neural_data, locs_cm, min_spikes = 100):
    neural_data = abs(neural_data)
    #Remove neurons with too few spikes in HC dataset
    nd_sum=np.nansum(neural_data,axis=0) #Total number of spikes of each neuron
    rmv_nrn=np.where(nd_sum<min_spikes) #Find neurons who have less than 100 spikes total
    neural_data=np.delete(neural_data,rmv_nrn,1) #Remove those neurons
    v
    X=neural_data
    print(X.shape)
    
    y = locs_cm
    #Remove time bins with no output (y value)
    rmv_time=np.where(X.sum(axis=1) == 0)
    X=np.delete(X,rmv_time,0)
    y=np.delete(y,rmv_time,0)
    
    print(y.shape)
    print(X.shape)
    return X, y

#
def prepare_data3(X, y, split=0.8):

    len_rec = X.shape[0]
    print ("length of moving period: ", len_rec/20., "sec")

    idx = int(len_rec*split)

    #
    X_train = X[:idx]
    X_test = X[idx:]
    y_train = y[:idx]
    y_test = y[idx:]

    return X_train, y_train, X_test, y_test



def prepare_data(X,y, neural_data, split=0.8):
    #X_train = X[initial:end,:]
    #y_train = y[initial:end,:]

    # find which neurons do not have any spikes
    rmv_nrn=np.where(X.sum(axis=0) == 0)
    X=np.delete(X,rmv_nrn,1) #Remove those neurons
    
    #
    neural_data = np.delete(neural_data,rmv_nrn,1).astype(int)
    return X, X_train, y_train, neural_data


def split_data(X,y, neural_data, initial=0, end=10000):
    X_train = X[initial:end,:]
    y_train = y[initial:end,:]

    rmv_nrn=np.where(X_train.sum(axis=0) == 0)
    X_train=np.delete(X_train,rmv_nrn,1) #Remove those neurons
    X=np.delete(X,rmv_nrn,1) #Remove those neurons
    neural_data = np.delete(neural_data,rmv_nrn,1).astype(int)
    return X, X_train, y_train, neural_data

def preprocess_bayesian_validation(neural_data, locs_cm, min_spikes = 100):

    X=neural_data
    print(X.shape)
    
    y = locs_cm
    #Remove time bins with no output (y value)
    rmv_time=np.where(X.sum(axis=1) == 0)
    X=np.delete(X,rmv_time,0)
    y=np.delete(y,rmv_time,0)
    
    print(y.shape)
    print(X.shape)
    return X, y


NaiveBayesDecoder = NaiveBayesRegression

def compute_occupancy(times):
    occupancy = [len(sublist) for sublist in times]
    norm_occupancy = [item/(sum(occupancy)) for item in occupancy]
    return norm_occupancy

def compute_firing_rate(f_binned, times):
    firing_rates = []
    for cell in range(f_binned.shape[1]):
        firing_rates.append([])
        for position in range(len(times)):
            r_i = sum(f_binned[:,cell][times[position]])
            firing_rates[cell].append(r_i)
    return firing_rates

def normalised_firing_rate(f_binned, times, norm_occupancy):
    firing_rates = []
    for cell in range(f_binned.shape[1]):
        firing_rates.append([])
        for position in range(len(times)):
            r_i = sum(f_binned[:,cell][times[position]])/norm_occupancy[position]
            firing_rates[cell].append(r_i)
    return firing_rates

def compute_place_cellness(firing_rates, nom_occupancy):
    place_cellness = []
    for cell in range(len(firing_rate)):
        numerator = sum((np.array(firing_rates[cell])*np.array(norm_occupancy))**2)
        denominator = sum((np.array(firing_rates[cell])**2)*np.array(norm_occupancy))
        place_cellness.append(numerator/denominator)

    bin_place_cellness = (np.array(place_cellness) > 0.3).astype(int)
    return place_cellness

def filter_times(partition_times, idx):
    mob_times = []
    for i in range(len(partition_times)):
        indices=np.argwhere(np.isin(partition_times[i], idx))[:,0]
        mob_times.append(partition_times[i][indices])
    return mob_times

def run_pca_return_pca(data):
    from sklearn.decomposition import PCA
    pca = PCA(svd_solver='full')
    X_pca = pca.fit_transform(data)
    
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    print(cumsum)
    #print ("cumsum: ", cumsum)
    idx = np.where(cumsum>0.95)[0]
    
    return pca, X_pca[:,:idx[0]]


def fit_data(X_training_data, y_training_data, dimensionality):
        
    # reduce dimensionality by taking only the first X cells or PCs
    # for now we should leave all the dimensions in the data
    X_training_data = X_training_data[:,:dimensionality]
    
    # compute average neural state at each spatial location
    # this can be done on all data or PCA version up to some dimensionality
    ave_states = []
    loc_idx = []
    for i in range(int(max(y_training_data))):
        idx = np.where(y_training_data == i)
        loc_idx.append(idx)

    for k in range(len(loc_idx)):
        temp = X_training_data[loc_idx[k]]
        #print ("temp: ", temp.shape)
        ave = np.nanmedian(temp,axis=0)
        ave_states.append(ave)
        
    data_triaged_ave = np.vstack(ave_states)

	# The rest of this is for network/graph analysis
    # we now have the average neural activity representing eachlocation in space:
    #  will build transition matrix for each neural state occuring at time t
    #    1) identify which location on track is nearest at time=t
    #    2) identify which location on track is nearest to time=t+1

  
    return data_triaged_ave

def predict_data(data_triaged_ave, X_test_data, dimensionality):
    
    
    neural_location = np.zeros(X_test_data.shape[0],dtype='int32')
    #print ("data tiraged average: ", data_triaged_ave)
    for k in range(X_test_data.shape[0]-1):

        # find location of current state in PCA-space
        temp_state = X_test_data[k,:dimensionality]  # grab only first n comps
        diff = np.abs(data_triaged_ave-temp_state).sum(axis=1)
        idx1 = np.nanargmin(diff)
        neural_location[k] = idx1

    return neural_location


# predict bayes
def predict_bayes_stationary(root_dir,
                            animal_id,
                            session_id,
                            use_place_cells,
                            overwrite=False,
                            shuffle=False,
                            speed_threshold=2):
    #
    fname_out = os.path.join(root_dir,
                             animal_id,
                             session_id,
                             "prediction_stationary_place_cells_"+str(use_place_cells)+'.npz')

    if os.path.exists(fname_out) == False or overwrite == True:

        fname_csv = glob.glob(os.path.join(root_dir,
                                           animal_id,
                                           session_id,
                                           "*0.csv"))[0]
        #
        fname_locs = fname_csv[:-4] + '_locs.npy'

        #
        fname_bin = glob.glob(os.path.join(root_dir,
                                           animal_id,
                                           session_id,
                                           'binarized_traces.npz'))[0]

        #
        print ("TODO: use start offset for mouse 7050: ")
        fname_start = os.path.join(root_dir,
                                   animal_id,
                                   session_id,
                                   'start.txt')

        # Get the data in bins
        partition_size = 10

        (f_binned,
         locs_partitioned,
         partition_times,
         partition,
         box_width,
         box_length,
         locs_cm,
         partition_size,
         ) = get_data_decoders_2D(fname_locs,
                                  fname_bin,
                                  partition_size=partition_size)

        # trim longer results files [THIS SHOULD EVENTUALLY BE DONE PRE PROCESSING]
        if f_binned.shape[0] < locs_partitioned.shape[0]:
            #locs_partitioned = locs_partitioned[locs_partitioned.shape[0] - f_binned.shape[0]:]
            locs_cm = locs_cm[locs_cm.shape[0] - f_binned.shape[0]:, :]

        #
        neural_data = f_binned.copy()

        #############################################
        #############################################
        #############################################
        place_cells_dict = np.load(os.path.join(os.path.split(fname_locs)[0], 'place_cells.npy'), allow_pickle=True)
        ids = []
        for c in place_cells_dict:
            ids.append(c['cell_id'])
        #
        place_cell_ids = np.hstack(ids)
        #
        if use_place_cells:
            neural_data = neural_data[:,place_cell_ids]

        #############################################
        #############################################
        #############################################
        # COMPUTE THE SPEED
        # speed_threshold = 2 #Set a threshold for the speed
        #speed = compute_speed(locs_cm, smooth=True, frames_smooth=20)
        #mobile, idx_mob, idx_imm = is_mobile(speed, threshold=speed_threshold)


        #############################################
        #############################################
        #############################################
        bayes_trainer = np.load(os.path.join(root_dir,
                                animal_id,
                                session_id,
                                "bayes_decoder_place_cells_"+str(use_place_cells)+'.npz'),
                                allow_pickle=True)

        active_cells = bayes_trainer['active_cells']
        idx_imm = bayes_trainer['idx_imm']
        idx_mob = bayes_trainer['idx_mob']
        x_imm = np.int32(neural_data[idx_imm])
        y_imm = locs_cm[idx_imm]
        print("Time moving: ", idx_mob.shape[0] / 20., ", time not moving: ", idx_imm.shape[0] / 20.)


        ########## LOAD THE TRAINER ##########
        fname_bayes = os.path.join(root_dir,
                                    animal_id,
                                    session_id,
                                    "bayes_decoder_place_cells_"+str(use_place_cells)+'.pkl')

        import pickle
        def pickle_loader(filename):
            """ Deserialize a file of pickled objects. """
            with open(filename, "rb") as f:
                while True:
                    try:
                        yield pickle.load(f)
                    except EOFError:
                        break

        for k in pickle_loader(fname_bayes):
            model_nb = k

        ########################################################################
        ########################################################################
        ########################################################################
        x_in = x_imm[:, active_cells]
        y_test = y_imm[:]

        #
        if shuffle:
            y_in = np.roll(y_test, active_times.shape[0]//2,axis=0)

        #
        y_predicted = model_nb.predict(x_in, y_test)

        # Get the distance between the predicted value and the true value
        distance_error = get_distance(y_test, y_predicted)

        #
        if shuffle==False:
            np.savez(fname_out,
                     distance_error = distance_error,
                     place_cell_ids = place_cell_ids,
                     all_cell_ids = np.arange(neural_data.shape[1]),
                     neural_data_shape = neural_data.shape,
                     y_test = y_test,
                     y_predicted = y_predicted,
                     active_cells = active_cells,
                     idx_mob = idx_mob,
                     idx_imm = idx_imm)
        else:
            print ("Skipping save for shuffle condition...")
        print ('')

    else:
        d = np.load(fname_out)
        distance_error = d['distance_error']

    return distance_error


class Volition():

    def __init__(self, root_dir, animal_id):
        self.root_dir = root_dir
        self.animal_id = animal_id

        self.session_ids = [
            "FS1",
            "FS2",
            "FS3",
            "FS4",
            "FS5",
            "FS6",
            "FS7",
            "FS8",
            "FS9",
            "FS10",
            "FS11",
            "FS12"
        ]

    def plot_grid(self, figsize=None):
        
        #
        d = np.load(os.path.join(self.root_dir,
                                 self.animal_id,
                                 self.session_id,
                                 'bayes_decoder_place_cells_False.npz'),
                    allow_pickle=True)

        #
        partition = d['partition']
        print(partition)

        #
        par_size = d['partition_size']
        print(par_size)

        #
        if figsize is None:
            fig = plt.figure()
        else:
            fig = plt.figure(figsize=figsize)

        #
        for k in range(partition.shape[0] + 1):
            # for p in range(partition.shape[1]):
            plt.plot([0, 80], [k * par_size, k * par_size], c='black')

        for k in range(partition.shape[1] + 1):
            # for p in range(partition.shape[1]):
            plt.plot([k * par_size, k * par_size], [0, 80], c='black')

        #
        plt.show()

        return fig


    def get_matches_forward(self):

        # find distance between predicted and true
        # focus primarily on perdicted distances that are further than some min distance

        if self.before_running:
            pass

        #
        dists = np.linalg.norm(self.imm_pred - self.imm_true, axis=1)
        idx = np.where(dists > self.min_dist_difference)[0]

        #
        if self.shuffle_match:

            self.match_times = []
            self.all_times = []
            self.match_array = []
            self.hist_array = []
            self.dists_array = []
            self.match_times_array = []
            for k in trange(self.n_shuffles):
                matches = 0
                total_times = 0
                match_times = []
                dist_array = []
                all_times = []
                locs_cm_r = np.roll(self.locs_cm, np.random.choice(np.arange(1000, 10000, 1), 1), axis=0).copy()

                #
                for id_ in idx:

                    #
                    loc_current = self.imm_true[id_]

                    #
                    loc_predicted = self.imm_pred[id_]

                    #
                    #dist_now = np.linalg.norm(loc_predicted - loc_current)
                    #if abs(dist_now) < self.min_dist_difference:
                    #    continue

                    # find correct time back in stack
                    abs_time = self.idx_imm_abs[id_]

                    #
                    locs_forward = locs_cm_r[abs_time:abs_time + self.n_sec * self.fps]

                    # check distances between future locations and prediction
                    dists = np.linalg.norm(loc_predicted - locs_forward, axis=1)

                    #
                    argmin = np.argmin(dists)

                    #
                    if dists[argmin] < self.min_dist_approached:
                        matches += 1
                        match_times.append(argmin)
                        dist_array.append(np.linalg.norm(loc_current - loc_predicted))
                    #else:
                    total_times += 1
                    all_times.append(argmin)

                #
                match_times = np.array(match_times) / self.fps
                all_times = np.array(all_times)/self.fps
                y = np.histogram(match_times, bins=np.arange(0, self.n_sec, self.bin_width))
                ym = np.histogram(all_times, bins=np.arange(0, self.n_sec, self.bin_width))
                self.hist_array.append((y[0]+1)/(ym[0]+1))
                self.match_array.append(matches)
                #self.match_times_array.append(match_times)
                #self.dists_array.append(dist_array)
                #self.match_times.append(match_times)
                self.all_times.append(all_times)

            #
            self.match_times = np.mean(np.vstack(self.match_array), axis=0)
            self.all_times = np.mean(np.vstack(self.all_times), axis=0)
            y = np.histogram(self.match_times, bins=np.arange(0, self.n_sec, self.bin_width))
            ya = np.histogram(self.all_times, bins=np.arange(0, self.n_sec, self.bin_width))

            self.hist = np.mean(np.vstack(self.hist_array), axis=0)
            self.std = np.std(np.vstack(self.hist_array), axis=0)

            self.x = y[1][:-1]
            self.match_ids = None

        else:
            #
            self.matches = 0
            self.total_times = 0
            self.match_times = []
            all_times = []
            self.match_ids = []
            self.dists_array = []
            for id_ in idx:

                #
                loc_current = self.imm_true[id_]

                #
                loc_predicted = self.imm_pred[id_]

                #
                #dist_now = np.linalg.norm(loc_predicted-loc_current)
                #print ("dist now: ", dist_now)
                #if abs(dist_now)<self.min_dist_difference:
                 #   continue


                # find correct time back in stack
                abs_time = self.idx_imm_abs[id_]

                #
                locs_forward = self.locs_cm[abs_time:abs_time + self.n_sec * self.fps]

                # check if they are closer
                dists = np.linalg.norm(loc_predicted - locs_forward, axis=1)

                #
                argmin = np.argmin(dists)

                if dists[argmin] < self.min_dist_approached:
                    self.matches += 1
                    self.match_times.append(argmin)
                    self.match_ids.append(id_)
                    self.dists_array.append(np.linalg.norm(loc_current - loc_predicted))

                #
                self.total_times += 1
                all_times.append(argmin)

            #
            self.match_times = np.array(self.match_times) / self.fps
            self.all_times = np.array(all_times) / self.fps
            y = np.histogram(self.match_times, bins=np.arange(0, self.n_sec, self.bin_width))
            ym = np.histogram(self.all_times, bins=np.arange(0, self.n_sec, self.bin_width))
            self.hist = (y[0]+1)/(ym[0]+1)
            self.x = y[1][:-1]
            self.match_times_array = self.match_times
            self.std=np.zeros(self.hist.shape[0])
        #return matches, match_times, hist, x, match_ids, dists_array, match_times_array



    def get_matches_forward_backward(self):

        # get datasets
        self.imm_pred, self.imm_true, self.idx_imm_abs, self.locs_cm = get_prediction_data(self.root_dir,
                                                                       self.animal_id,
                                                                       self.session_id,
                                                                       self.mobile
                                                                       )

        if self.direction == "forward":
            self.get_matches_forward()
        else:
            self.get_matches_backward()


    #
    def plot_decoder_errors(self):

        ls = []
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.session_ids)))
        dists_m = []
        dists_imm = []
        for i, session_id in enumerate(self.session_ids):
            fname_out = os.path.join(self.root_dir,
                                     self.animal_id,
                                     session_id,
                                     "bayes_decoder_place_cells_" + str(self.use_place_cells_bayes) + '.npz')
            data = np.load(fname_out, allow_pickle=True)
            dist_m = data['distance_error_mobile']
            dist_im = data['distance_error_immobile']
            y_m = np.histogram(dist_m, bins=np.arange(0, 120, 1))
            y_imm = np.histogram(dist_im, bins=np.arange(0, 120, 1))

            dists_m.append(y_m[0])
            dists_imm.append(y_imm[0])

        fig = plt.figure()
        x = y_m[1][:-1]
        ax = plt.subplot(1, 1, 1)
        mean = np.array(dists_m).mean(0)
        std = np.std(np.array(dists_m), axis=0)

        plt.plot(mean, c='blue', label='moving')
        ax.fill_between(x, mean - std, mean + std,
                        color='blue', alpha=.25)
        ax.legend(loc=1)
        plt.ylim(bottom=0)
        plt.xlim(x[0], x[-1])
        ax.set_ylabel("Moving periods")
        ##########################
        ax2 = ax.twinx()
        mean = np.array(dists_imm).mean(0)
        std = np.std(np.array(dists_imm), axis=0)

        plt.plot(mean, c='red', label='stationary')
        ax2.fill_between(x, mean - std, mean + std,
                         color='red', alpha=.25)

        ax2.legend(loc=2)
        ax2.set_ylabel("Stationary periods")
        plt.ylim(bottom=0)
        plt.xlim(x[0], x[-1])
        ax.set_xlabel("Error (cm)")
        plt.title(self.animal_id)

        #
        plt.show()

    #
    def convert_locations(self):

        #
        for session_id in tqdm(self.session_ids):

            self.fname_csv = glob.glob(os.path.join(self.root_dir,
                                               self.animal_id,
                                               session_id,
                                               "*0.csv"))[0]

            fname_out = self.fname_csv[:-4] + "_locs.npy"
            if os.path.exists(fname_out):
                continue

            #
            locs = load_csv(fname_csv)
            body_feature_idx = 5 * 3 + 1

            # print ("Body feature index", np.arange(body_feature_idx,body_feature_idx+2,1))
            neck = np.float32(locs[3:, body_feature_idx:body_feature_idx + 2])
            np.save(fname_csv[:-4] + '_locs.npy', neck)



    def animate_movement_open_field(self,frame):
        #global ctr, fig, ax1, ax2, sizes2

        return

        #
        plt.suptitle("Time: "+str(round(ctr/sample_rate,2)).zfill(2))

        ########### TRACK SPACE #########
        ax1.clear()
        ax1=plt.subplot(1,2,1)

        #sizes = np.zeros(res.shape[0])+25
        #idx = pos_track[ctr]
        #idx = np.random.choice(np.arange(180),1)
        #sizes[idx]=250
        #ax1.imshow(partition)
        draw_grid(ax1)

        x = locs_cm[ctr,0]/partition_size
        y = locs_cm[ctr,1]/partition_size
        ax1.scatter(x,
                    y,
                    c='blue',
                    s=200)
        ax1.set_title("Mouse location x,y "+str(round(x*partition_size,1))+ " "+str(round(y*partition_size,1)) +
                    " (cm)")

        ax1.set_xlim(0,box_width//partition_size)
        ax1.set_ylim(0,box_length//partition_size)

        ax1.set_xticks([])
        ax1.set_yticks([])

        ############################################
        ########### NEURAL STATE SPACE #############
        ############################################
        ax2.clear()
        ax2=plt.subplot(1,2,2)
        ax2.set_title("Neural state inferred location")
        #sizes = np.zeros(G.number_of_nodes())+25

        draw_grid(ax2)
        temp = neural_location[ctr]
        x = temp//(box_width//partition_size)+0.5
        y = temp%(box_length//partition_size)+0.5
        
        #
        ax2.scatter(x,
                    y,
                    c='red',
                    s=200)
        plt.xlim(0,box_width//partition_size)
        plt.ylim(0,box_length//partition_size)

        ax2.set_xticks([])
        ax2.set_yticks([])
        
        if ctr%50==0:
            print ("ctr: ", ctr)

        #
        ctr+=1

        
    #
    def make_movies(self):

        #
        from matplotlib import animation


        # get location of the mouse
        self.locs_cm 
        print ("locs_cm: ", self.locs_cm.shape)

        # indexes of the immobile periods
        self.idx_imobile_absolute #= #idx_imm[active_times_immobile],  # this is the relative times of the stationary times
        print ("idx_imobile_absolute: ", self.idx_imobile_absolute.shape)

        # locations of predicted immobile periods
        self.y_immobile_predicted 
        print ("y_immobile_predicted: ", self.y_immobile_predicted.shape)
        
        # note used for now
        # self.y_mobile_predicted,

        # indexes of the mobile periods
        self.idx_mob 
        print ("idx_mob: ", self.idx_mob.shape)
        self.idx_imm
        print ("idx_imm: ", self.idx_imm.shape)

        #
        figsize=(6,6)
        self.fig = self.plot_grid(figsize)


        ####                          
        ani = animation.FuncAnimation(self.fig, 
                                      self.animate_movement_open_field, 
                                      #frames=neural_location.shape[0]-1
                                      frames=50,
                                      interval=1, 
                                      repeat=True)
        
        #
        fname = '/home/cat/test_vid.mp4'
        ani.save(fname, 
                #writer='imagemagick', 
                fps=20)
        ani.close()


    # Iterate through all the sessions
    def train_bayes(self,
                    speed_threshold=2):
        
        #
        for session_id in self.session_ids:
            
            #
            fname_out = os.path.join(self.root_dir,
                                     self.animal_id,
                                     session_id,
                                     "bayes_decoder_place_cells_"+str(self.use_place_cells_bayes)+'.npz')
            
            #
            if os.path.exists(fname_out) == False or self.overwrite_bayes == True:

                fname_csv = glob.glob(os.path.join(self.root_dir,
                                                   self.animal_id,
                                                   session_id,
                                                   "*0.csv"))[0]
                #
                fname_locs = fname_csv[:-4] + '_locs.npy'

                #
                temp = os.path.join(self.root_dir,
                                    self.animal_id,
                                       session_id,
                                       '*binarized_traces*.npz')
                #print ("temp: ", temp)
                fname_bin = glob.glob(temp)[0]

                #
                print ("TODO: use start offset for mouse 7050: ")
                fname_start = os.path.join(self.root_dir,
                                           self.animal_id,
                                           session_id,
                                           'start.txt')

                # Get the data in bins
                partition_size = 10

                (f_binned,
                 locs_partitioned,
                 partition_times,
                 partition,
                 box_width,
                 box_length,
                 locs_cm,
                 partition_size,
                 ) = get_data_decoders_2D(fname_locs,
                                          fname_bin,
                                          partition_size=partition_size)

                # trim longer results files [THIS SHOULD EVENTUALLY BE DONE PRE PROCESSING]
                if f_binned.shape[0] < locs_partitioned.shape[0]:
                    #locs_partitioned = locs_partitioned[locs_partitioned.shape[0] - f_binned.shape[0]:]
                    locs_cm = locs_cm[locs_cm.shape[0] - f_binned.shape[0]:, :]

                #
                neural_data = f_binned.copy()

                #############################################
                #############################################
                #############################################
                place_cells_dict = np.load(os.path.join(os.path.split(fname_locs)[0], 'place_cells.npy'), allow_pickle=True)
                ids = []
                for c in place_cells_dict:
                    ids.append(c['cell_id'])
                #
                place_cell_ids = np.hstack(ids)
                #
                if self.use_place_cells_bayes:
                    neural_data = neural_data[:,place_cell_ids]

                #############################################
                #############################################
                #############################################
                # COMPUTE THE SPEED
                # speed_threshold = 2 #Set a threshold for the speed
                self.speed = compute_speed(locs_cm, smooth=True, frames_smooth=20)
                mobile, idx_mob, idx_imm = is_mobile(self.speed, threshold=speed_threshold)

                #
                x_mob = np.int32(neural_data[idx_mob])
                y_mob = locs_cm[idx_mob]
                print("Time moving: ", idx_mob.shape[0] / 20., ", time not moving: ", idx_imm.shape[0] / 20.)

                #############################################
                #############################################
                #############################################
                split = 0.9
                print("total time, total cells: ", x_mob.shape)
                X_train, y_train, X_mobile_test, y_mobile_test = prepare_data3(x_mob, y_mob, split)
                active_times_mobile, active_cells = find_active_times_and_cells(X_train,
                                                                         min_spikes=25)

                #
                X_in = X_train[active_times_mobile][:, active_cells]
                y_in = y_train[active_times_mobile]

                #
                if self.shuffle_bayes:
                    y_in = np.roll(y_in, active_times_mobile.shape[0]//2,axis=0)

                ###############################################
                ############### TRAIN ON MOVING ###############
                ###############################################
                model_nb = NaiveBayesRegression(res=10)
                model_nb.fit(X_in, y_in)

                ###############################################
                ############### PREDICT ON MOVING #############
                ###############################################
                # TODO: prediction fucntion should not get ground truth!
                active_times_test = get_active_frames(X_mobile_test[:,active_cells])
                y_mobile_predicted = model_nb.predict(X_mobile_test[active_times_test][:, active_cells],
                                                      y_mobile_test[active_times_test])

                # Get the distance between the predicted value and the true value
                distance_error_mobile = get_distance(y_mobile_test[active_times_test],
                                                     y_mobile_predicted)

                ###############################################
                ############# PREDICT ON IMMOBILE #############
                ###############################################
                x_immobile = np.int32(neural_data[idx_imm])  # neural data
                y_immobile_test = locs_cm[idx_imm]           # location data
                active_times_immobile = get_active_frames(x_immobile)  # remove frames with no spikes

                # predict
                y_immobile_predicted = model_nb.predict(x_immobile[active_times_immobile][:, active_cells],
                                                        y_immobile_test[active_times_immobile]  # this is not necessary
                                                        )
                
                # compute error
                distance_error_immobile = get_distance(y_immobile_test[active_times_immobile],
                                                       y_immobile_predicted)

                ###############################################
                ###############################################
                ###############################################
                if self.shuffle_bayes==False:
                    np.savez(fname_out,
                             distance_error_mobile = distance_error_mobile,
                             distance_error_immobile = distance_error_immobile,
                             
                             # this is the relative times of the stationary times
                             # so indexes into all data to finde immobile periods: idx_imm
                             #               then searches the active frames: active_times_immobile
                             idx_imobile_active_absolute = idx_imm[active_times_immobile],  

                            #
                             locs_cm = locs_cm,
                             neural_data = neural_data,

                             #
                             place_cell_ids = place_cell_ids,
                             all_cell_ids = np.arange(neural_data.shape[1]),
                             neural_data_shape = neural_data.shape,
                             y_immobile_predicted = y_immobile_predicted,
                             y_mobile_predicted = y_mobile_predicted,
                             active_cells = active_cells,
                             active_times_mobile = active_times_mobile,
                             active_times_test = active_times_test,
                             idx_mob = idx_mob,
                             idx_imm = idx_imm,
                             speed = self.speed,
                             split = split,

                             #
                             partition = partition,
                             partition_times = partition_times,
                             partition_size = partition_size,
                             locs_partitioned = locs_partitioned)

                    with open(fname_out[:-4]+'.pkl', 'wb') as outp:
                        #company1 = Company('banana', 40)
                        pickle.dump(model_nb, outp, pickle.HIGHEST_PROTOCOL)

                else:
                    print ("Skipping save for shuffle condition...")
                print ('')

            #
            else:

                self.load_bayes()


    def load_bayes(self):
        
        #
        fname_out = os.path.join(self.root_dir,
                                     self.animal_id,
                                     self.session_id,
                                     "bayes_decoder_place_cells_"+str(self.use_place_cells_bayes)+'.npz')
        
        #
        d = np.load(fname_out, allow_pickle=True)
        
        #
        self.distance_error_mobile = d['distance_error_mobile']
        self.distance_error_immobile = d['distance_error_immobile']
        self.idx_imobile_active_absolute = d['idx_imobile_active_absolute']
        self.locs_cm = d['locs_cm']
        self.neural_data = d['neural_data']
        self.place_cell_ids = d['place_cell_ids']
        self.all_cell_ids = d['all_cell_ids']
        self.neural_data_shape = d['neural_data_shape']
        self.y_immobile_predicted = d['y_immobile_predicted']
        self.y_mobile_predicted = d['y_mobile_predicted']
        self.active_cells = d['active_cells']
        self.active_times_mobile = d['active_times_mobile']
        self.active_times_test = d['active_times_test']
        self.idx_mob = d['idx_mob']
        self.idx_imm = d['idx_imm']
        self.speed = d['speed']
        self.split = d['split']
        self.partition = d['partition']
        self.partition_times = d['partition_times']
        self.partition_size = d['partition_size']
        self.locs_partitioned = d['locs_partitioned']

    #
    def get_bouts(self, idx_imm):
        # detect bouts:
        imms = []
        start = idx_imm[0]
        for k in range(1,idx_imm.shape[0]-1,1):
            if idx_imm[k]!=(idx_imm[k-1]+1):
                imms.append([start,idx_imm[k-1]])
                start = idx_imm[k]

        imms = np.vstack(imms)

        return imms






def get_matches_backward(n_sec,
                        fps,
                        min_dist,
                        imm_pred,
                        imm_true,
                        idx_imm_abs,
                        locs_cm,
                        shuffle,
                        bin_width=0.25,
                        ):
    # dists = scipy.spatial.distance.cdist(imm_pred,
    #                                         imm_true, metric='euclidean')
    dists = np.linalg.norm(imm_pred - imm_true, axis=1)

    idx = np.where(dists > min_dist)[0]
    #print(idx.shape, '/', dists.shape)

    #
    if shuffle:

        match_array = []
        hist_array = []
        for k in range(10):
            matches = 0
            match_times = []
            locs_cm_r= np.roll(locs_cm, np.random.choice(np.arange(1000,10000,1),1), axis=0).copy()

            for id_ in tqdm(idx):

                #
                loc_current = imm_tru[id_]

                #
                loc_predicted = imm_pred[id_]

                # find correct time back in stack
                abs_time = idx_imm_abs[id_]

                #
                locs_backward = locs_cm_r[abs_time - n_sec * fps: abs_time]

                # check if they are closer
                try:
                    dists = np.linalg.norm(locs_backward - loc_predicted, axis=1)
                    argmin = np.argmin(dists)
                except:
                    continue

                #

                if dists[argmin] < min_dist:
                    matches += 1
                    match_times.append(-argmin)
                    dist_array.append(np.linalg.norm(loc_current - loc_predicted))
            #
            match_times = np.array(match_times) / fps
            y = np.histogram(match_times, bins=np.arange(-n_sec,0, bin_width))
            hist_array.append(y[0])
            match_array.append(matches)
            dists_array.append(dist_array)

        #
        match_times = np.mean(np.vstack(match_array),axis=0)
        hist = np.mean(np.vstack(hist_array),axis=0)[::-1]
        x = y[1][:-1]
        match_ids = None

    else:
        #
        matches = 0
        match_times = []
        match_ids = []
        dists_array = []
        for id_ in tqdm(idx):

            #
            loc_current = imm_true[id_]

            #
            loc_predicted = imm_pred[id_]

            # find correct time back in stack
            abs_time = idx_imm_abs[id_]

            #
            locs_backward = locs_cm[abs_time - n_sec * fps: abs_time]

            # check if they are closer
            try:
                dists = np.linalg.norm(locs_backward - loc_predicted, axis=1)
                argmin = np.argmin(dists)
            except:
                continue
            #

            if dists[argmin]<min_dist:
                matches+=1
                match_times.append(-argmin)
                match_ids.append(id_)
                dists_array.append(np.linalg.norm(loc_current-loc_predicted))

        #
        match_times = np.array(match_times)/fps
        y = np.histogram(match_times, bins=np.arange(-n_sec,0, bin_width))
        hist = y[0][::-1]
        x=y[1][:-1]

    return matches, match_times, hist, x, match_ids, dists_array



def get_prediction_data(root_dir,
                        animal_id,
                        session_id,
                        mobile
                        ):
    
    #
    data = np.load(os.path.join(root_dir,
                                animal_id,
                                session_id,
                                'bayes_decoder_place_cells_False.npz'),
                   allow_pickle=True)

    # get decoded frames with no mobility and at least some [ca] activity
    # do analysis over immobile periods
    if mobile == False:
        idx_imm_abs = data['idx_imobile_absolute']
        print("idx immobile periods: ", idx_imm_abs.shape)
        imm_pred = data['y_immobile_predicted']

    # do future movement analysis over moving periods
    else:
        # here we have to carefully double index into data
        # TODO: check this at some point...
        #
        idx_mob = data['idx_mob']

        #
        active_times = data['active_times_test']
        print("non-zero [ca], i.e. active frames: ", active_times.shape)
        idx_imm_abs = idx_mob[active_times]

        print("idx mobile periods: ", idx_imm_abs.shape)
        imm_pred = data['y_mobile_predicted']

    #
    neural_data, locs_cm = get_neural_data_and_locs(root_dir,
                                                    animal_id,
                                                    session_id
                                                    )
    #
    imm_true = locs_cm[idx_imm_abs]
    print("imm true: ", imm_true.shape)

    return imm_pred, imm_true, idx_imm_abs, locs_cm
