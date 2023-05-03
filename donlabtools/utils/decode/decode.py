import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm, trange
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from scipy.stats import norm
from scipy.spatial.distance import cdist
import math
import seaborn as sns
import pickle
import statsmodels.api as sm


#

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
            #if True:
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


#
def get_active_frames(neural_data, min_activity = 1):

    #print ("min spikes: ", min_spikes)
    # Remove time bins with no neural activity
    good_times=np.where(neural_data.sum(axis=1) >= min_activity)[0]

    return good_times

    
def get_distance(y, y_pred):
    x_dec = y_pred[:,0]
    y_dec = y_pred[:,1]
    x_cm = y[:,0]
    y_cm = y[:,1]
    distance = np.sqrt((x_dec-x_cm)**2+(y_dec-y_cm)**2)
    return distance

   
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


#
def bayesian_decoding(ani):
    
    #
    fname_out = os.path.join(ani.root_dir,
                             ani.animal_id,
                        ani.session_id,
                        "bayes_decoder_place_cells_"+str(ani.use_place_cells)+'.npz')


    # trim longer results files [THIS SHOULD EVENTUALLY BE DONE PRE PROCESSING]
    if ani.f_binned.shape[0] < ani.locs_partitioned.shape[0]:
        #locs_partitioned = locs_partitioned[locs_partitioned.shape[0] - f_binned.shape[0]:]
        ani.locs_cm = ani.locs_cm[ani.locs_cm.shape[0] - seanilf.f_binned.shape[0]:, :]

    #
    #############################################
    if ani.use_place_cells:
        place_cells_dict = np.load(os.path.join(os.path.split(ani.fname_locs)[0], 'place_cells.npy'), allow_pickle=True)
        ids = []
        for c in place_cells_dict:
            ids.append(c['cell_id'])
        #
        place_cell_ids = np.hstack(ids)
        #
        ani.neural_data_local = ani.neural_data[:,place_cell_ids].copy()
    else:
        place_cell_ids = None
        ani.neural_data_local = ani.neural_data.copy()

    #############################################
    #############################################
    #############################################
    # COMPUTE THE SPEED
    # speed_threshold = 2 #Set a threshold for the speed
    ani.compute_speed()

    #
    x_mob = np.int32(ani.neural_data_local[ani.idx_mob])
    y_mob = ani.locs_cm[ani.idx_mob]
    print("Time moving: ", ani.idx_mob.shape[0] / 20., 
            ", time not moving: ", ani.idx_imm.shape[0] / 20.)

    #############################################
    #############################################
    #############################################
    print("total time, total cells: ", x_mob.shape)
    X_train, y_train, X_mobile_test, y_mobile_test = prepare_data3(x_mob, y_mob, 
                                                                    ani.data_split)
    active_times_mobile, active_cells = find_active_times_and_cells(X_train,
                                                                    min_spikes=25)

    #
    X_in = X_train[active_times_mobile][:, active_cells]
    y_in = y_train[active_times_mobile]
    ani.y_train = y_in
    
    #
    if ani.shuffle:
        y_in = np.roll(y_in, active_times_mobile.shape[0]//2,axis=0)

    ##################################
    ########### TRAIN MODEL ##########
    ##################################
    model_nb = NaiveBayesRegression(res=10)
    print ("X_in.shape: ", X_in.shape)
    print ("y_in.shape: ", y_in.shape)
    model_nb.fit(X_in, y_in)

    ########################################
    ############ TEST MOBILE DATA ##########
    ########################################
    # TODO: prediction fucntion should not get ground truth!
    active_times_test = get_active_frames(X_mobile_test[:,active_cells])
    y_mobile_test = y_mobile_test[active_times_test]
    X_mobile_test = X_mobile_test[active_times_test][:,active_cells]

    # check if the test data is within 3 cm of the training data
    if ani.remove_nonvisited_locs:
        idx_remove = []
        for k in range(y_mobile_test.shape[0]):
            loc = y_mobile_test[k]
            dist = np.linalg.norm(loc-ani.y_train,axis=1)
            cl = np.min(dist)
            if cl>3:
                print ("test removed from training set because too far ", cl , " cm")
                idx_remove.append(k)

        # remove the bad indices
        y_mobile_test = np.delete(y_mobile_test, idx_remove, axis=0)
        X_mobile_test = np.delete(X_mobile_test, idx_remove, axis=0)

    y_mobile_predicted = model_nb.predict(X_mobile_test,
                                            y_mobile_test)

    # Get the distance between the predicted value and the true value
    distance_error_mobile = get_distance(y_mobile_test,
                                            y_mobile_predicted)

    ###############################################
    ############# PREDICT IMMOBILE ###############
    ###############################################
    if ani.predict_immobile:
        x_immobile = np.int32(ani.neural_data_local[ani.idx_imm])
        y_immobile_test = ani.locs_cm[ani.idx_imm]
        active_times_immobile = get_active_frames(x_immobile)

        #
        y_immobile_predicted = model_nb.predict(x_immobile[active_times_immobile][:, active_cells],
                                                y_immobile_test)
        distance_error_immobile = get_distance(y_immobile_test[active_times_immobile],
                                                y_immobile_predicted)
    else:
        distance_error_immobile = np.nan
        y_immobile_predicted = np.nan
        active_times_immobile = []

    if ani.plot_bayesian_decoding:
        fig = plt.figure()
        ax = sns.violinplot(data = distance_error_mobile, 
                            showfliers=ani,
                            #showmeans=True,
                            showmedians=ani,)
        plt.ylabel("Distance error (cm)")
        plt.xlabel("Session")
        plt.show()

    ###############################################
    ###############################################
    ###############################################
    ani.distance_error_mobile = distance_error_mobile
    ani.distance_error_immobile = distance_error_immobile

    if ani.shuffle==False:
        np.savez(fname_out,
                    distance_error_mobile = distance_error_mobile,
                    distance_error_immobile = distance_error_immobile,
                    idx_imobile_active_absolute = ani.idx_imm[active_times_immobile],  # this is the relative times of the stationary times
                    locs_cm = ani.locs_cm,
                    neural_data = ani.neural_data,

                    #
                    place_cell_ids = place_cell_ids,
                    all_cell_ids = np.arange(ani.neural_data.shape[1]),
                    neural_data_shape = ani.neural_data.shape,
                    y_immobile_predicted = y_immobile_predicted,
                    y_mobile_predicted = y_mobile_predicted,
                    active_cells = active_cells,
                    active_times_mobile = active_times_mobile,
                    active_times_test = active_times_test,
                    idx_mob = ani.idx_mob,
                    idx_imm = ani.idx_imm,
                    speed = ani.speed_threshold,
                    split = ani.data_split,

                    #
                    partition = ani.partition,
                    partition_times = ani.partition_times,
                    partition_size = ani.partition_size,
                    locs_partitioned = ani.locs_partitioned)

        with open(fname_out[:-4]+'.pkl', 'wb') as outp:
            #company1 = Company('banana', 40)
            pickle.dump(model_nb, outp, pickle.HIGHEST_PROTOCOL)

    else:
        print ("Skipping save for shuffle condition...")
    print ('')

def prepare_data3(X, y, split=0.8):

    len_rec = X.shape[0]
    print ("length of moving period: ", len_rec/20., "sec")

    idx = int(len_rec*split)

    #
    X_train = X[:idx]
    X_test = X[idx:]
    y_train = y[:idx]
    y_test = y[idx:]

    #
    return X_train, y_train, X_test, y_test


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