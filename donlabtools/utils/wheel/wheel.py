import numpy as np
import os
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def convert_polar_to_xy(track_pos, sample_rate):
    # 
    n_cycles_per_rotation = 500
    track_circular = (track_pos % n_cycles_per_rotation) * 360 / n_cycles_per_rotation

    # 
    x, y = pol2cart(1, track_circular)

    return x, y


class empty:

    def __init__(self):
        pass

class attribs:
    pass

#
class Wheel():

    #
    def __init__(self):

        self.sample_rate = 10000
        self.imaging_sample_rate = 30
        # print ("Wheel sample rate: ", self.sample_rate, " hz")

        # initialize a wheel subobject
        self.wheel = empty()
        self.wheel.radius = 0.1 # wheel radius in meters
        self.wheel.clicks_per_rotation = 500
        self.wheel.length_of_click = (2*np.pi*self.wheel.radius/self.wheel.clicks_per_rotation)


        # initialize a track subobject
        self.track = empty()

    #
    def load_track(self):

        self.fname = os.path.join(self.root_dir,'wheel.npy')

        self.track.rotary_binarized = empty()
        self.track.rotary_binarized.values = -np.load(self.fname)

        # get time on track
        times = np.arange(self.track.rotary_binarized.values.shape[0]) / self.sample_rate
        self.track.rotary_binarized.times = times

        #
        self.track.positions = empty()
        self.track.positions.values = np.cumsum(self.track.rotary_binarized.values)
        self.track.positions.times = self.track.rotary_binarized.times

        #
        self.track.distances = empty()
        self.track.distances.values = self.track.positions.values* self.wheel.length_of_click
        self.track.distances.times = self.track.rotary_binarized.times

        #
        self.track.fname_galvo_trigger = os.path.join(self.root_dir, '2p_galvo_trigger.npy')

        # load galvo time stamps;
        # value > 0 for when 2P frame is obtained;
        # value = 0 in between

        # find rising edge of trigger
        fname_out = os.path.join(os.path.split(self.fname)[0],
                                 'triggers.npy')
        try:
            triggers = np.load(fname_out)
        except:
            ch4 = np.load(self.track.fname_galvo_trigger)
            idx = np.where(ch4 >= 1)[0]
            ch4[:] = 0
            ch4[idx] = 1

            #
            triggers = []
            for id_ in tqdm(idx):
                if ch4[id_-1]==0:
                    triggers.append(id_)

            triggers = np.array(triggers)
            np.save(fname_out, triggers)

        #
        self.track.galvo_triggers = empty()
        self.track.galvo_triggers.times = triggers


    def compute_xy(self):
    #
        print(track.shape)
        track_sum = np.cumsum(track)
        print("track sum: ", track_sum)

        #
        track_pos = np.float32(-track_sum)

        # convert to x,y coords along track - OPTIONAL
        x, y = convert_polar_to_xy(track_pos,
                                   sample_rate)
        
        self.x = x
        sefl.y = y

    def plot_track(self, obj, ax=None, c='black', label=None):

        #t = np.arange(self.track_velocity.shape[0])/self.sample_rate
        if ax is None:
            plt.figure()
            ax=plt.subplot(111)

        #compute_velocity
        if False:
            ax.plot(t, track, c='blue', label='Instantaneous velocity')
            ax.set_xlim(t[0], t[-1])


        #
        t = obj.times
        vals = obj.values
        ax.plot(t, vals, c=c,
                label=label)

        #
        ax.set_xlim(t[0], t[-1])
        ax.set_xlabel("Time (sec)", fontsize=20)
        ax.set_ylabel("Velocity (m/sec)", fontsize=20)

        #
        ax.legend(fontsize=10, loc=2)
        #ax.suptitle(self.fname)
        plt.show()

    def plot_track_3D(self):

        import plotly.express as px
        temp = np.array([x,y,t]).T
        df = pd.DataFrame(temp, columns = ['x','y','time'])

        print ("# of time steps: ", track_circular.shape)
        print (df)
        #
        fig = px.line_3d(df,
                           x='x',
                           y='time',
                           z='y'
                         #,
                          # width=1,
                           #height=1
                        )
        #fig = px.line_3d(df, x="gdpPercap", y="pop", z="year")
        fig.show()

            # 
    def get_indexes_run_periods(self):
        
        # select only quiescent periods
        idx = np.where(self.track.velocity.values>self.min_velocity_running)[0]

        print ("   Running periods (seconds): ", idx.shape[0]/self.imaging_sample_rate)
        
        return idx
    
    # 
    def get_indexes_quiescent_periods(self):
        
        # select only quiescent periods
        idx = np.where(self.track.velocity.values<=self.max_velocity_quiescent)[0]

        print ("   Queiscent periods (seconds): ", idx.shape[0]/self.imaging_sample_rate)
        
        return idx
        

    #
    def compute_velocity(self):

        fname_out = os.path.join(os.path.split(self.fname)[0],'velocity.npy')

        #
        # print ("LOADING: ", fname_out)
        #
        if os.path.exists(fname_out):
            galvo_vel = np.load(fname_out)

        else:

            #
            click_distance = ( 2 *np.pi *self.wheel.radius ) /self.wheel.clicks_per_rotation

            # convert rotary encoder clicks to velocity

            #
            v = np.zeros(self.track.rotary_binarized.values.shape[0], dtype=np.float32)
            idx = np.where(self.track.rotary_binarized.values!=0)[0]
            print ("non-zero rotary encoder vals: ", idx)

            #
            last_time_idx = 0
            for id_ in tqdm(idx):

                #
                click = self.track.rotary_binarized.values[id_]  # can be +/- 1

                #
                v[id_] = (click*click_distance ) /((id_ -last_time_idx ) /self.sample_rate)

                #
                last_time_idx = id_

            # remove all zeros
            idx2 = np.where(v == 0)[0]
            v2 = np.delete(v, idx2)

            # set velocity times based on non-zero rotatry encoder vals
            vel_times = np.int32(np.delete(self.track.rotary_binarized.times, idx2)*self.sample_rate)
            print ("vel_times: ", vel_times)

            # refill in the rotary encoder when it's in an undefined state for > 0.5 sec
            vel_full = []
            vel_full.append(v2[0])
            times_full = []
            times_full.append(vel_times[0])
            max_time = 5000 # in 10Khz sample rate
            for k in trange(1,vel_times.shape[0],1, desc='refilling stationary times'):
                if (vel_times[k]-vel_times[k-1])>=max_time:
                    temp = np.arange(vel_times[k-1], vel_times[k], max_time)
                    #print ("adding # zero times: ", temp.shape)
                    times_full.append(temp[1:])
                    vel_full.append(temp[1:]*0)
                else:
                    #print ('vel_times[k]', vel_times[k])
                    #print ("times full: ", times_full)
                    times_full.append(vel_times[k])
                    vel_full.append(v2[k])

            vel_times = np.hstack(times_full)
            v2 = np.hstack(vel_full)

            print ("vel times: ", vel_times.shape)
            print ("vel values: ", v2.shape, v2)

            # Fit the detected velocity times and values
            if v2.shape[0]!=0:
                F = interp1d(vel_times,
                             v2,
                             fill_value='extrapolate')

                print ("self.track.galvo_triggers.times: ", self.track.galvo_triggers.times)
                print ("vel_times:", vel_times)

                # resample the fit function at the correct galvo_tirgger times
                galvo_vel = F(self.track.galvo_triggers.times)
            else:
                galvo_vel = v*0


            np.save(fname_out, galvo_vel)

            #
        self.track.velocity = empty()
        self.track.velocity.values = galvo_vel
        self.track.velocity.times = self.track.galvo_triggers.times/self.sample_rate
