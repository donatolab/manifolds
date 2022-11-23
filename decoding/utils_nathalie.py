# Visualisation
import matplotlib.pyplot as plt

# Computing
import numpy as np
from scipy import io
from scipy import stats
import sys
import pickle


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
    print ("bin upphase: ", F_upphase.shape)

    # start of recording as some imaging didn't start same time as DLC tracking

    #
    locs = np.load(fname_locs)

    print ("All DLC locations: ", locs.shape)
    locs = locs[0:36000]
    print ("Final locations shape: ", locs.shape)
    F_upphase = F_upphase[0:36000]
    print ("bin upphase: ", F_upphase.shape)

    return (F_upphase, locs)

def get_data_decoders_2D(fname_locs,
             fname_bin, partition_size=5):

    F_upphase, locs = load_tracks_2D(fname_locs,
                                           fname_bin)
    print ("locs: ", locs.shape)
    # partition space into specific parcels
    F_upphase = F_upphase.T
    F_upphase = F_upphase[0:36000]
    print ("F: ", F_upphase.shape)

    # bin data
    bin_size = 7
    run_binnflag = False
    if run_binnflag:
        sum_flag = True
        f_binned= run_binning(F_upphase, bin_size, sum_flag)
    else:
        f_binned = F_upphase
    print (f_binned.shape)

    # 
    if run_binnflag:
        sum_flag = False
        locs_binned = run_binning(locs, bin_size, sum_flag)
    else:
        locs_binned = locs
    print ("locs_binned: ", locs_binned.shape)


    # partition space 
    box_width, box_length = 80, 80    # length in centimeters

    # centre to 0,0
    locs[:,0] = locs[:,0]-np.min(locs[:,0])
    locs[:,1] = locs[:,1]-np.min(locs[:,1])

    xmax = np.max(locs[:,0])
    ymax = np.max(locs[:,1])
    print (xmax,ymax)
    pixels_per_cm = xmax / box_width

    # convert DLC coords to cm
    locs_cm = locs_binned/pixels_per_cm
    print ("locs_cm: ", locs_cm.shape)

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


    print ("DONE")
    
    return f_binned, locs_partitioned, partition_times, partition, box_width, box_length, locs_cm,partition_size

    
    
def compute_speed(locs_cm, time_frame = 0.05, smooth = True, frames_smooth = 100):
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

    print ("Note... saving all locations regardless of likelihood...")
    print (locs.shape)

    return locs


from sklearn.decomposition import PCA

class ProjectionDecoder(object):

    """
    Class for the Nearest Neighbor Decoder
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
            try:
                #print ("neuron number: ", j)
                #print ("y_train: ", y_train)
                tuning=glm_run(y_train,X_b_train[:,j:j+1],input_xy)
            
                #Enter tuning curves into matrix
                tuning_all[j,:]=np.squeeze(tuning)
            except:
                print ("cell broke: ", j)

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
        #print(prob_dists)
        #print(dists)
        #print(prob_dists.shape)
        
        #Setting the same probability for all the places
        prob_dists =np.ones(prob_dists.shape)

        #Initializations
        loc_idx= np.argmin(cdist(y_test[0:1,:],input_xy)) #The index of the first location
        num_nrns=tuning_all.shape[0] #Number of neurons
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
    
def preprocess_bayesian(neural_data, locs_cm, min_spikes = 100):

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
