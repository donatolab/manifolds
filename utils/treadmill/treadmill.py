
from scipy.spatial import cKDTree
import h5py
import os
import numpy as np
from tqdm import tqdm, trange
from scipy import stats
import matplotlib.pyplot as plt

class Treadmill():

    def __init__(self,
                 experiment_id,
                 session_ids,
                 up_phase,
                 reorder,
                 session_names
                 ):

        self.experiment_id = experiment_id
        self.session_ids = session_ids
        self.up_phase = up_phase
        self.reorder = reorder
        self.session_names = session_names

    def show_cell_session_averages(self):

        # load default names
        fname_tracks, fnames_ca, segment_order = reload_names()

        #
        fname_tracks = fname_tracks[self.experiment_id]
        fname_ca = fnames_ca[self.experiment_id]
        segment_order = segment_order[self.experiment_id]
        print("Segment order: ", segment_order)

        #
        data_ca = np.load(fname_ca)
        print("raw ca: ", data_ca.shape)

        #
        # print (fname_tracks)
        bin_width = 5
        pos_tracks, idx_tracks = load_tracks2(fname_tracks, bin_width)

        #
        print("total # segments pos_tracks: ", len(pos_tracks))

        vels = []
        for k in range(len(pos_tracks)):
            vel = pos_tracks[k][1:] - pos_tracks[k][:-1]
            idx = np.where(vel < 0)[0]
            vel[idx] = vel[idx - 1]
            vel = vel * 30.

            vels.append(vel)

        #
        len_seg_cm = 1  # discretize belt position into 6 ~equal sections
        triage_value = 0.25  # remove xx% most outlier states

        #
        imgs = []
        ca_array = np.zeros((len(self.session_ids), data_ca.shape[0], len(pos_tracks[0])))
        for session_id in self.session_ids:

            #
            idx_track = idx_tracks[session_id]
            pos_track = pos_tracks[session_id]
            fname_track = fname_tracks[session_id]
            pos_track = np.int32(pos_track // len_seg_cm)

            #
            print('')
            print("pos track: ", pos_track, pos_track.shape, " , # unique vals: ", np.unique(pos_track).shape)
            print("idx track: ", idx_track, idx_track.shape)

            #
            print(fname_track)
            print("fname ca: ", fname_ca)
            if self.up_phase:
                # fname_ca = fname_ca.replace('upphase','filtered')
                ca = np.load(fname_ca)
                # print ("loaded ca: ", np.unique(ca))
                # ca = ca - np.mean(ca, axis=1)[:, None]
            else:
                data = np.load('/media/cat/4TB/donato/steffen/DON-004366/20210303/binarized_traces.npz',
                               allow_pickle=True)
                ca = data["F_filtered"]

            print("ca shape: ", ca.shape)

            #
            print("sement order: ", segment_order)

            # make session average per cell:

            cell_ids = np.arange(ca.shape[0])
            # cell_ids = np.arange(10)
            img = []
            ctrc = 0
            for cell_id in tqdm(cell_ids):
                arr = np.zeros(180)

                #
                f0 = np.max(ca[cell_id])
                if f0 != 0:

                    #
                    for k in range(idx_track.shape[0] - 1):
                        temp = ca[cell_id, idx_track[k]]
                        #temp = ca[cell_id, idx_track[k]:idx_track[k] + bin_width].mean()
                        # temp = temp - f0
                        arr[pos_track[k]] = arr[pos_track[k]] + temp
                        ca_array[session_id, ctrc, k] = temp

                    # smoof
                    width = 11
                    arr = np.convolve(arr, np.ones(width) / width, mode='same')

                    #
                    if arr.sum() > 0:
                        # normalize
                        arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

                ctrc += 1

                #
                img.append(arr)

            #
            img = np.array(img)

            # save stack before reordering cells
            imgs.append(img.copy())

            #
            maxes = np.argmax(img, axis=1)
            print("maxes: ", maxes.shape, maxes)
            idx_main = np.argsort(maxes)[::-1]

            img = img[idx_main]

            #
            print(np.nanmax(arr), np.nanmin(arr))

            plt.figure()
            # ax=plt.subplot(1,2,1)
            plt.imshow(img,
                       aspect='auto',
                       interpolation='None'
                       )

            # ax=plt.subplot(1,2,2)
            # plt.imshow(ca,
            #           aspect='auto')
            plt.ylabel("Cell id (Ordered by ~peak")
            plt.xlabel("Belt " + str(segment_order[session_id]) + " location (cm)")
            plt.title(os.path.split(os.path.split(fname_ca)[0])[0])
            plt.suptitle("Segment: " + segment_order[session_id])
            plt.show(block=False)

            print("DONE...")

        self.imgs = imgs
        self.vels = np.array(vels)
        self.ca_array = ca_array

        self.pos_tracks = pos_tracks

        self.idx_tracks = idx_tracks
        #
        #

    def compute_activity_maps(self, min_vel=2):

        ''' This only computes the average activity at each

        :param min_vel:
        :return:
        '''

        #
        idx = np.where(self.vels[0] > min_vel)[0]
        print('# moving times: ', idx.shape)

        #
        pos = np.int32(self.pos_tracks[0][idx])
        ca = self.ca_array[0][:, idx]

        n_frames_in_loc = np.zeros(180)
        ave = np.zeros((ca.shape[0], 180))

        #
        for k in range(pos.shape[0]):
            ave[:, pos[k]] = ave[:, pos[k]] + ca[:, k]
            n_frames_in_loc[pos[k]] += 1

        #
        ave = ave / n_frames_in_loc

        #
        for k in range(ave.shape[0]):
            max_ = np.nanmax(ave[k])
            min_ = np.nanmin(ave[k])
            if max_ == min_:
                pass
            else:
                ave[k] = (ave[k] - min_) / (max_ - min_)

        # reorder cells
        if self.reorder:
            idx = np.argmax(ave, axis=1)
            idx2 = np.argsort(idx)
            ave = ave[idx2]

        #
        self.rate_maps = ave

        #
        print ("rate maps: ", self.rate_maps.shape)

        ##############################
        ##############################
        ##############################
        plt.figure()
        plt.imshow(ave,
                   aspect='auto')
        plt.show()

    def compute_rate_maps(self):

        ''' This only computes the average activity at eac

        :param min_vel:
        :return:
        '''


        # find moving
        self.ca = []
        self.pos = []
        self.rate_maps_non_norm = []
        self.n_frames_in_loc = []
        self.rate_maps = []
        for session_id in self.session_ids:

            #
            print (len(self.vels), len(self.vels[0]))
            idx = np.where(self.vels[session_id] > self.min_vel)[0]
            print('# moving times: ', idx.shape)

            #
            pos = np.int32(self.pos_tracks[session_id][idx])
            self.pos.append(pos)

            #
            ca = self.ca_array[session_id][:, idx]
            self.ca.append(ca)

            n_frames_in_loc = np.zeros(180)
            ave = np.zeros((ca.shape[0], 180))

            # do time binned averages by adding activity at eveyr position
            for k in range(pos.shape[0]):
                ave[:, pos[k]] = ave[:, pos[k]] + ca[:, k]
                n_frames_in_loc[pos[k]] += 1

            # smooth the
            width = 5
            if False:
                n_frames_in_loc = np.convolve(n_frames_in_loc, np.ones(width) / width, mode='same')

            #
            self.n_frames_in_loc.append(n_frames_in_loc)
            self.rate_maps_non_norm.append(ave.copy())

            #
            ave = ave / n_frames_in_loc

            #
            for k in range(ave.shape[0]):
                max_ = np.nanmax(ave[k])
                min_ = np.nanmin(ave[k])
                if max_ == min_:
                    pass
                else:
                    ave[k] = (ave[k] - min_) / (max_ - min_)

            #
            self.rate_maps.append(ave)

            #
            print ("rate maps: ", self.rate_maps[session_id].shape)

            ##############################
            ##############################
            ##############################
            plt.figure()
            # reorder cells
            if self.reorder_for_vis:
                idx = np.argmax(ave, axis=1)
                idx2 = np.argsort(idx)
                ave = ave[idx2]

            plt.imshow(ave,
                       aspect='auto')
            plt.title(self.session_names[session_id])
            plt.xlabel("Track (cm)")
            plt.ylabel("Cell #")

            plt.show(block=False)

    def compute_si(self):


        #
        self.si = np.zeros((len(self.ca),self.rate_maps_non_norm[0].shape[0]))
        self.si_rate = np.zeros((len(self.ca), self.rate_maps_non_norm[0].shape[0]))
        self.zscore = np.zeros((len(self.ca),self.rate_maps_non_norm[0].shape[0]))

        for session_id in self.session_ids:
            #
            time_map = self.pos[session_id]
            print("time_map :", time_map.shape)
            for c in trange(self.rate_maps_non_norm[session_id].shape[0]):

                #
                #rate_map = self.rate_maps_non_norm[c]
                rate_map = self.ca[session_id][c]

                #
                inf_rate, inf_content = get_si2(rate_map, time_map)

                #
                self.si_rate[session_id, c] = inf_rate
                self.si[session_id, c] = inf_content

                # get zscore:
                si_shuffle = []
                for k in range(self.n_tests):
                    time_map2 = np.roll(time_map, np.random.choice(np.arange(time_map.shape[0]),1))
                    inf_rate, _ = get_si(rate_map, time_map2)
                    si_shuffle.append(inf_rate)

                from scipy import stats
                stack = np.hstack([self.si_rate[session_id, c], si_shuffle])
                self.zscore[session_id, c] = stats.zscore(stack)[0]
                #print (zz[0])

    def show_zscore_examples(self):

        ###################
        idx = np.where(np.logical_and(self.zscore[self.session_id] < 0.5, self.zscore[self.session_id] > -0.5))[0]
        plt.figure()
        plt.suptitle(self.session_names[self.session_id])
        ax = plt.subplot(3, 1, 1)
        plt.title("low zscores")
        width = 11
        for k in range(10):
            try:
                temp = self.rate_maps_non_norm[self.session_id][idx[k]]
                temp = np.convolve(temp, np.ones(width) / width, mode='same')
                plt.plot(temp / self.n_frames_in_loc[self.session_id] + k * 0.3)
                plt.plot(temp * 0 + k * 0.3, '--', c='grey')
            except:
                pass

        ##########################
        ax = plt.subplot(3, 1, 2)
        idx = np.where(self.zscore[self.session_id] > self.std_threshold)[0]
        plt.title("> " + str(self.std_threshold) + " std")
        for k in range(10):
            try:
                temp = self.rate_maps_non_norm[self.session_id][idx[k]]
                temp = np.convolve(temp, np.ones(width) / width, mode='same')
                plt.plot(temp / self.n_frames_in_loc[self.session_id] + k * 0.3)
                plt.plot(temp * 0 + k * 0.3, '--', c='grey')
            except:
                pass

        ##############################
        ax = plt.subplot(3, 1, 3)
        idx = np.where(self.zscore[self.session_id] < -self.std_threshold)[0]
        plt.title("< -" + str(self.std_threshold) + " std")
        for k in range(10):
            try:
                temp = self.rate_maps_non_norm[self.session_id][idx[k]]
                temp = np.convolve(temp, np.ones(width) / width, mode='same')
                plt.plot(temp / self.n_frames_in_loc[self.session_id] + k * 0.3)
                plt.plot(temp * 0 + k * 0.3, '--', c='grey')
            except:
                pass
        plt.xlabel("Belt (cm)")
        plt.show(block=False)

    def zscore_si_histogram(self):
        # std_threshold = 2.5

        idx = np.where(np.abs(self.zscore[self.session_id]) >= self.std_threshold)[0]
        print ("idx: ", idx)

        plt.figure()
        plt.title("Zscore distributions - Seg " + str(self.session_names[self.session_id]) + " , std threshold " + str(
            self.std_threshold))
        plt.hist(self.zscore[self.session_id], label="# place cells: " + str(idx.shape[0]) + ", " +
                                  str(int(idx.shape[0] / self.zscore.shape[0] * 100)) + "%")
        plt.ylabel("# cells")
        plt.xlabel("Zscore")
        plt.legend()
        plt.show(block=False)

    def compute_pairse_correlation_distributions(imgs,
                                                 session_ids,
                                                 offset,
                                                 unscramble,
                                                 pval_thresh,
                                                 corr_thresh,
                                                 scramble_order,
                                                 shuffle=False,
                                                 ):
        #
        sess1 = imgs[session_ids[0]].copy()
        sess2 = imgs[session_ids[1]].copy()

        #
        corrs = []
        pvals = []
        locs = []
        peaks = []
        changes = []
        stacks = []
        ctr = 0

        # loop over all cells
        for k in range(sess1.shape[0]):

            # load the cell
            seg1 = sess1[k]
            #
            seg2 = sess2[k]
            if shuffle:
                # print (seg2.shape)
                # print(np.arange(5000, seg2.shape[0]-5000,1))
                seg2 = np.roll(seg2,
                               np.random.choice(np.arange(0, 180, 1), 1)
                               # 60
                               )

            #
            if unscramble:
                temp = seg2.copy()
                for k in range(6):
                    # print (k, k+30, scramble_order[k], scramble_order[k]+30)
                    seg2[k * 30:k * 30 + 30] = temp[scramble_order[k] * 30:scramble_order[k] * 30 + 30]

            #
            try:
                res = stats.pearsonr(seg1[offset:], seg2[offset:])
            except:
                continue

            #
            corr = res[0]
            pval = res[1]

            #
            if pval < pval_thresh and abs(corr) > corr_thresh:
                corrs.append(res[0])
                pvals.append(res[1])

                #
                loc = np.argmax(seg1[offset:]) + offset
                locs.append(loc)

                #
                peak1 = np.argmax(seg1[offset:]) + offset
                peak2 = np.argmax(seg2[offset:]) + offset
                peaks.append([peak1,
                              peak2])

                #
                changes.append(peak1 - peak2)

                #
                if corr > corr_thresh:
                    seg1 = seg1[offset:]
                    seg2 = seg2[offset:]
                    stacks.append((seg1 - np.min(seg1)) / (np.max(seg1 - np.min(seg1))))
                    stacks.append((seg2 - np.min(seg2)) / (np.max(seg2 - np.min(seg2))))
                    stacks.append(seg1 * 0)

                #
                ctr += 1
        #
        return np.array(corrs), np.array(locs), peaks, changes, stacks

    def show_rigid_cells_from_correlations(self):
        idx = np.where(np.abs(self.zscore[self.session_id]) >= self.std_threshold)[0]
        print(idx.shape)

        # grab the rate maps
        r1 = self.rate_maps[0][idx]
        print(r1.shape)

        r2 = self.rate_maps[2][idx]
        print(r2.shape)

        #
        corrs = []
        pvals = []
        for k in range(r1.shape[0]):
            res = stats.pearsonr(r1[k], r2[k])
            corrs.append(res[0])
            pvals.append(res[1])
        pvals = np.array(pvals)
        corrs= np.array(corrs)

        plt.figure()
        n_place = idx.shape[0]
        plt.suptitle(" Total # place cells: " + str(n_place) + " in session A")
        y = np.histogram(corrs, bins=np.arange(-1, 1, .05))
        plt.bar(y[1][:-1], y[0], width=0.04, label='non-rigid')

        #
        idx = np.where(np.abs(y[1][:-1]) >= 0.3)[0]
        print("idx: ", idx)
        plt.bar(y[1][:-1][idx], y[0][idx], width=0.04, label='rigid (pcorr>0.3) :' + str(y[0][idx].sum()))
        plt.legend()
        plt.xlabel("Pearson corr")
        plt.ylabel("# cells")
        plt.show(block=False)

        ##############################################
        plt.figure()
        y = np.histogram(pvals, bins=np.arange(0, 1, 0.05))
        plt.bar(y[1][:-1],y[0], width=.04)
        plt.xlabel("Pval")
        plt.ylabel("# cells")
        plt.show(block=False)

        plt.figure()
        idx = np.where(pvals<0.05)[0]
        print (idx)
        y = np.histogram(corrs[idx], bins=np.arange(-1, 1, .05))
        plt.bar(y[1][:-1], y[0], width=0.04, label='statistically significant cell-pairs '+str(idx.shape[0]) +
                ", " + str(int(idx.shape[0]/n_place*100))+ "%")
        plt.xlabel("Pearson corr")
        plt.legend()
        plt.ylabel("# cells")
        plt.title("Cells with statistically signifcant relationships")
        plt.show(block=False)



    def get_cell_peak_shifts2(self
                             ):
        #
        sess1 = self.rate_maps[self.session_ids[0]]
        sess2 = self.rate_maps[self.session_ids[1]]
        #self.rate_maps

        # first select only cells above some zscore
        idx = np.where(np.abs(self.zscore[0]) >= self.std_threshold)[0]
        print("# of cells selected: ", idx.shape)

        sess1 = sess1[idx]
        sess2 = sess2[idx]


        #
        corrs = []
        pvals = []
        locs = []
        peaks = []
        changes = []
        stacks = []
        ctr = 0

        # loop over all cells
        for k in range(sess1.shape[0]):

            # load the cell
            seg1 = sess1[k]
            #
            seg2 = sess2[k]
            #print (seg1.shape, seg2.shape)

            if self.shuffle:
                # print (seg2.shape)
                # print(np.arange(5000, seg2.shape[0]-5000,1))
                seg2 = np.roll(seg2,
                               np.random.choice(np.arange(0, 180, 1), 1)
                               # 60
                               )

            #
            if self.unscramble:
                temp = seg2.copy()
                for k in range(6):
                    # print (k, k+30, scramble_order[k], scramble_order[k]+30)
                    seg2[k * 30:k * 30 + 30] = temp[scramble_order[k] * 30:scramble_order[k] * 30 + 30]

            #

            res = stats.pearsonr(seg1, seg2)
            #except:
            #    continue

            #
            #print (k, "res: ", res)
            corr = res[0]
            pval = res[1]

            #
            if (pval <= self.pval_threshold) and (abs(corr) >= self.corr_threshold):
                #print ("iiming")
                corrs.append(res[0])
                pvals.append(res[1])

                #
                loc = np.argmax(seg1)
                locs.append(loc)

                #
                peak1 = np.argmax(seg1)
                peak2 = np.argmax(seg2)
                peaks.append([peak1,
                              peak2])

                #
                changes.append(peak1 - peak2)

                #
                if corr > self.corr_threshold:
                    seg1 = seg1
                    seg2 = seg2
                    stacks.append((seg1 - np.min(seg1)) / (np.max(seg1 - np.min(seg1))))
                    stacks.append((seg2 - np.min(seg2)) / (np.max(seg2 - np.min(seg2))))
                    stacks.append(seg1 * 0)


                #
                ctr += 1

        #
        self.locs = locs
        self.peaks = peaks
        self.changes = np.array(changes)
        self.stacks = stacks
        self.corrs = corrs
        print ("# of cells with canges: ", self.changes.shape)




    def make_circular_plots2(self):
        #

        print ("self.changes: ", self.changes)
        changes2 = np.array(self.changes)
        print ("changes2: ", changes2)

        # unrap the distributions
        idx = np.where(changes2 < 0)[0]
        changes2[idx] += 180

        # map360 degrees to 180 degrees; basiclaly each 45 degree is half that
        y = np.histogram(changes2, bins=np.arange(0, 180 + self.width, self.width))

        # plt.show(block=False)
        xx = y[1][:-1]  # *(360/150)
        yy = y[0]

        ######################################
        plt.figure(figsize=(10, 6))
        plt.subplot(polar=True)

        theta = np.linspace(0, 2 * np.pi, xx.shape[0])

        # Arranging the grid into number
        # of sales into equal parts in
        # degrees
        lines, labels = plt.thetagrids(range(0, 360, int(360 / len(yy))),
                                       (xx))

        # Plot actual sales graph
        theta = np.hstack((theta, [theta[0]]))
        yy = np.hstack((yy, [yy[0]]))
        plt.plot(theta, yy, linewidth=3, label="Remapping distance")
        # plt.fill(theta, yy, 'b', alpha = 0.1)
        plt.legend()
        # Plot expected sales graph
        # plt.plot(theta, expected)
        plt.suptitle(self.session_names[self.session_ids[0]] + " vs " + self.session_names[self.session_ids[1]] +
                     " , # cells: " + str(len(self.corrs)))
        # Display the plot on the screen
        plt.show(block=False)

        #####################################################
        #####################################################
        #####################################################
        peaks = np.vstack(self.peaks)
        plt.figure()
        # plt.scatter(peaks[:,0],peaks[:,1])
        # width = 30
        y = np.histogram(changes2, bins=np.arange(0, 180 + self.width, self.width))
        # plt.plot(y[1][:-1], y[0])
        plt.bar(y[1][:-1], y[0], self.width * .9)

        plt.xlabel("Change in peak location for cells w. correlation pvalue < " + str(self.pval_threshold) + " (cm)")
        # plt.xlim(0,180)
        plt.show(block=False)


    def show_scatter_plots(imgs,
                           session_ids,
                           offset,
                           unscramble,
                           pval_thresh,
                           corr_thresh,
                           shuffle,
                           names):

        #
        corrs, locs, peaks, changes, stacks = compute_pairse_correlation_distributions(imgs,
                                                                                       session_ids,
                                                                                       offset,
                                                                                       unscramble,
                                                                                       pval_thresh,
                                                                                       corr_thresh,
                                                                                       shuffle)

        #
        plt.figure()
        idx1 = np.where(corrs > 0)[0]
        idx2 = np.where(corrs < 0)[0]
        plt.scatter(locs[idx1], corrs[idx1], c='blue')
        plt.scatter(locs[idx2], corrs[idx2], c='red')

        #
        corrs = np.array(corrs)
        locs = np.array(locs)
        #
        idx1 = np.where(corrs > 0)[0]
        bin_width = 5
        y1 = np.histogram(locs[idx1], bins=np.arange(0, 180 + bin_width, bin_width))
        plt.plot(y1[1][:-1] + bin_width / 2.,
                 y1[0] / np.max(y1[0]) + 0.2, c='blue', label='Pos corr')

        #
        idx2 = np.where(corrs < 0)[0]
        y2 = np.histogram(locs[idx2], bins=np.arange(0, 180 + bin_width, bin_width))
        plt.plot(y2[1][:-1] + bin_width / 2.,
                 -y2[0] / np.max(y2[0]) - 0.2, c='red', label='Neg corr')

        #
        plt.xlabel("Belt location of peak activation")
        plt.ylabel("Pearson corr")
        plt.xlim(0, 180)
        plt.title(
            "Pcorr >" + str(corr_thresh) + ", # of cells with pval <" + str(pval_thresh) + " : " + str(len(corrs)))
        plt.legend()
        plt.suptitle(names[session_ids[0]] + " vs " + names[session_ids[1]] + " , # cells: " + str(len(corrs)))
        plt.show(block=False)

    def visualize_matched_cells(stacks,
                                offset,
                                corr_thresh
                                ):
        stacks_out = np.array(stacks)
        print(stacks_out.shape)
        #
        subsample = 3
        maxes = np.argmax(stacks_out[::subsample, offset:], axis=1)
        print("maxes: ", maxes.shape)
        idx = np.argsort(maxes)[::-1]
        print(idx)

        temp = []
        for k in range(idx.shape[0]):
            temp.append(stacks_out[idx[k] * subsample])
            temp.append(stacks_out[idx[k] * subsample + 1])
            temp.append(stacks_out[idx[k] * subsample] * 0)

        #
        stacks_out = np.array(temp)

        # stacks_out = np.array(stacks)
        # print (stacks.shape)
        plt.figure()
        plt.imshow(stacks_out[:, offset:],
                   interpolation="none",
                   extent=[offset, 180, stacks_out.shape[0], offset],
                   aspect='auto')
        plt.ylabel("Pairs of cells extracted by correlation")
        plt.xlabel("Belt location (excluding first 20cm")
        plt.title("Pairs of cells (cell from seg 1, seg 2, space) with >" + str(corr_thresh) + " correlation")
        plt.suptitle("A vs. A'")
        plt.tight_layout()

        plt.show(block=False)


def get_si2(rate_map,
            time_map):
    duration = np.ma.sum(time_map)
    position_PDF = time_map / (duration + np.spacing(1))

    mean_rate = np.ma.sum(rate_map * position_PDF)
    mean_rate_sq = np.ma.sum(np.ma.power(rate_map, 2) * position_PDF)

    max_rate = np.max(rate_map)

    #if mean_rate_sq != 0:
    #    sparsity = mean_rate * mean_rate / mean_rate_sq

    #if mean_rate != 0:
    selectivity = max_rate / mean_rate

    log_argument = rate_map / mean_rate
    log_argument[log_argument < 1] = 1


    inf_rate = np.ma.sum(position_PDF * rate_map * np.ma.log2(log_argument))
    inf_content = inf_rate / mean_rate

    return inf_rate, inf_content

def get_si(rate_map,
           time_map):
    #
    duration = np.ma.sum(time_map)
    position_PDF = time_map / (duration + np.spacing(1))

    #
    # mean_rate = np.ma.sum(rate_map * position_PDF)
    mean_rate = np.sum(rate_map * position_PDF)

    #
    log_argument = rate_map / mean_rate
    log_argument[log_argument < 1] = 1

    #
    #inf_rate = np.ma.sum(position_PDF * rate_map * np.ma.log2(log_argument))
    inf_rate = np.sum(position_PDF * rate_map * np.log2(log_argument))
    inf_content = inf_rate / mean_rate

    return inf_rate, inf_content


def reload_names():
    fname_tracks = [

        [
            '/media/cat/4TB/donato/steffen/DON-004366/20210228/suite2p/plane0/mat/DON-004366_20210228_TRD-2P_S1-ACQ_eval.mat',
            '/media/cat/4TB/donato/steffen/DON-004366/20210228/suite2p/plane0/mat/DON-004366_20210228_TRD-2P_S2-ACQ_eval.mat',
            '/media/cat/4TB/donato/steffen/DON-004366/20210228/suite2p/plane0/mat/DON-004366_20210228_TRD-2P_S3-ACQ_eval.mat'],

        #
        ['/media/cat/4TB/donato/steffen/DON-004366/20210301/mat/DON-004366_20210301_TRD-2P_S1-ACQ_eval.mat',
         '/media/cat/4TB/donato/steffen/DON-004366/20210301/mat/DON-004366_20210301_TRD-2P_S2-ACQ_eval.mat',
         '/media/cat/4TB/donato/steffen/DON-004366/20210301/mat/DON-004366_20210301_TRD-2P_S3-ACQ_eval.mat'],

        #
        ['/media/cat/4TB/donato/steffen/DON-004366/20210303/DON-004366_20210303_TRD-2P_S1-ACQ_eval.mat',
         '/media/cat/4TB/donato/steffen/DON-004366/20210303/DON-004366_20210303_TRD-2P_S2-ACQ_eval.mat',
         '/media/cat/4TB/donato/steffen/DON-004366/20210303/DON-004366_20210303_TRD-2P_S3-ACQ_eval.mat']

    ]

    #
    fnames_ca = [
        '/media/cat/4TB/donato/steffen/DON-004366/20210228/suite2p/plane0/binarized_traces/F_upphase.npy',
        '/media/cat/4TB/donato/steffen/DON-004366/20210301/binarized_traces/F_upphase.npy',
        '/media/cat/4TB/donato/steffen/DON-004366/20210303/binarized_traces/F_upphase.npy']

    # segmaentaiton:
    segment_order = [
        ["A", "B", "A'"],
        ["A", "A'", "B"],
        ["A", "B", "A'"]
    ]

    return fname_tracks, fnames_ca, segment_order



def load_tracks2(fname_tracks, bin_width):
    #
    pos_tracks = []
    idx_tracks = []
    idx_ctr = 0

    # loop over all 3 segments here while loading
    for fname_track in fname_tracks:
        with h5py.File(fname_track, 'r') as file:
            #
            pos = file['trdEval']['position_atframe'][()].squeeze()

            n_timesteps = pos.shape[0]
            # print ("# of time steps: ", pos.shape)

        # bin speed for some # of bins
        # bin_width = 1
        sum_flag = False
        pos_bin = run_binning(pos, bin_width, sum_flag)
        # print ("pos bin: ", pos_bin.shape)

        #
        pos_tracks.append(pos_bin)

        # add locations of the belt realtive
        temp = np.arange(idx_ctr, idx_ctr + n_timesteps, 1)
        temp = np.int32(run_binning(temp, bin_width, sum_flag))
        idx_tracks.append(temp)

        idx_ctr += n_timesteps

    return pos_tracks, idx_tracks




#
def run_binning(data, bin_size=7, sum_flag=True):
    # split data into bins
    idx = np.arange(0, data.shape[0], bin_size)
    d2 = np.array_split(data, idx[1:])

    # sum on time axis; drop last value
    if sum_flag:
        d3 = np.array(d2[:-1]).sum(1)
    else:
        d3 = np.median(np.array(d2[:-1]), axis=1)

    print("   Data binned using ", bin_size, " frame bins, final size:  ", d3.shape)

    #
    return d3