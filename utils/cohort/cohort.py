import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm, trange


class Cohort():

    def __init__(self, mouse_ids, PDays, Wheel):

        #
        self.fps = 30

        #
        self.clrs = ['black','blue','red','green']

        #
        self.animal_ids = mouse_ids

        #
        self.PDays = PDays

        #
        self.Wheel = Wheel

    def plot_recording_summary(self):
        #
        fig = plt.figure()
        ax = plt.subplot()
        xlabels = self.animal_ids

        for k in range(len(self.animal_ids)):
            # y = np.arange(PDays[k][0],PDays[k][1],0.1)
            x = np.arange(self.PDays[k][0], self.PDays[k][1], 0.1)
            y = k
            # plt.fill_betweenx(y, k, k+0.9,
            # alpha=.5, label = mouse_ids[k])

            # x = np.arange(
            alpha = 1.0
            if k > 2:
                alpha = .3
            ax.fill_between(x, k, k + 0.9, label=self.animal_ids[k],
                            alpha=alpha)

            plt.text(np.median(x) - 3, k + 0.3, str(self.PDays[k][1] - self.PDays[k][0] + 1))

        plt.xlim(left=0)
        plt.ylabel("mice ids")
        plt.yticks(np.arange(len(self.animal_ids)) + 0.5, self.animal_ids,
                   rotation=45)
        plt.xlabel("P-Day")
        plt.legend(ncol=2, fontsize=8, loc=4)
        plt.show()


    def show_n_cell_allmice(self):
    #
        fig = plt.figure(figsize=(10, 5))
        ax = plt.subplot()

        for k in range(len(self.animal_ids)):
            animal_id = self.animal_ids[k]

            #
            root_dir = os.path.join('/media/cat/4TB1/donato/',
                                    animal_id)

            #
            sessions = np.sort(os.listdir(root_dir))

            #
            n_cells = []
            for session in sessions:

                # load the iscell value from suite2p output
                fname = os.path.join(root_dir, session,
                                     '002P-F/tif/suite2p/plane0/iscell.npy')

                #
                data = np.load(fname)
                idx = np.where(data[:, 0] == 1)[0]
                n_cells.append(idx.shape[0])

            #
            x = np.arange(self.PDays[k][0], self.PDays[k][1], 1)

            #
            plt.plot(x,
                     n_cells,
                     c=self.clrs[k],
                     linewidth=3,
                     label=self.animal_ids[k])

        #
        plt.title("# cells based on 'iscell' suite2p parameter")

        #
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.ylabel("# of cells ", fontsize=16)

        #
        plt.xlabel("P-Day", fontsize=16)
        plt.legend(ncol=2)
        plt.show()


    def show_n_frames_allmice(self):

        #
        fig = plt.figure(figsize=(10, 5))
        ax = plt.subplot()

        for k in range(len(self.animal_ids)):
            animal_id = self.animal_ids[k]

            #
            root_dir = os.path.join('/media/cat/4TB1/donato/',
                                    animal_id)

            #
            sessions = np.sort(os.listdir(root_dir))

            #
            n_frames = []
            for session in sessions:
                #
                fname = os.path.join(root_dir, session,
                                     '002P-F/tif/suite2p/plane0/F.npy')

                #
                # data = np.load(fname)
                data = np.load(fname, mmap_mode='r+')
                # print (data.shape)
                n_frames.append(data.shape[1] / self.fps)

            #
            x = np.arange(self.PDays[k][0], self.PDays[k][1], 1)

            #
            plt.plot(x,
                     n_frames,
                     c=self.clrs[k],
                     linewidth=3,
                     label=self.animal_ids[k])

        #
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.ylabel("# of seconds of recording ", fontsize=16)

        #
        plt.xlabel("P-Day", fontsize=16)
        plt.legend(ncol=2)
        plt.show()


    def show_distance_travelled_allmice(self):

        fig = plt.figure(figsize=(10, 5))
        ax = plt.subplot()


        for k in trange(len(self.animal_ids), desc="loading mice"):
            animal_id = self.animal_ids[k]

            #
            root_dir = os.path.join('/media/cat/4TB1/donato/',
                                    animal_id)

            #
            sessions = np.sort(os.listdir(root_dir))

            #
            distances = []
            for session in sessions:
            #

                w = self.Wheel()
                w.root_dir = os.path.join(root_dir, session, 'TRD-2P/')

                #
                w.load_track()

                #
                w.compute_velocity()

                # the track distance values should contain a cumulative sum over time
                distances.append(w.track.distances.values[-1])


            #
            x = np.arange(self.PDays[k][0], self.PDays[k][1], 1)

            #
            plt.plot(x,
                     distances,
                     c=self.clrs[k],
                     linewidth=3,
                     label=self.animal_ids[k])

          #  break

        #
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.ylabel("Distnace travelled (metres) ", fontsize=16)

        #
        plt.xlabel("P-Day", fontsize=16)
        plt.legend(ncol=2)
        plt.show()











