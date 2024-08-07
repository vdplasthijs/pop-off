## General imports (also for subsequent analysis notebooks)
import sys, os
import json

from popoff import loadpaths

print(loadpaths.__file__)

USER_PATHS_DICT = loadpaths.loadpaths()  # automatically use current PC user name 

path_to_vape = USER_PATHS_DICT['vape_path']
print(path_to_vape)

sys.path.append(str(path_to_vape))
sys.path.append(str(os.path.join(path_to_vape, 'jupyter')))
sys.path.append(str(os.path.join(path_to_vape, 'utils')))

oasis_path = USER_PATHS_DICT['oasis_path']
sys.path.append(str(oasis_path))
#print([x for x in sys.path if type(x) is not str])
sys_path_list = sys.path.copy()  # ensure all itmes in sys.path are strs (to prevent failure with core packages that use str only operations)
for x in sys_path_list:
    if type(x) is not str:
        sys.path.remove(x)
        if str(x) not in sys.path:
            sys.path.append(str(x))

import numpy as np
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import time, datetime
import utils_funcs as utils 
import run_functions as rf
from subsets_analysis import Subsets
import pickle
import sklearn.decomposition
from cycler import cycler

# OASIS gives a useless warning
import warnings
warnings.simplefilter("ignore", UserWarning)

from IPython.core.debugger import Pdb
ipdb = Pdb()
# Taken out to prevent need to clone OASIS to use SessionLite
#from oasis.functions import deconvolve

plt.rcParams['axes.prop_cycle'] = cycler(color=sns.color_palette('colorblind'))


def get_trial_frames_single(clock, start, pre_frames, post_frames, fs=30, paq_rate=20000):
    """Get frames that indicate trial start.

    Parameters:
    ----------------
        clock: ??
            ??
        start: ??
            ??
        pre_frames: int
            number of frames before stim to include in trial
        post_frames: int
            number of frames after stim to include in trial
        fs: int, default=30
            frame rate of imaging
        paq_rate: int, default=20000
            ??

    Returns:
    ---------------
        trial_frames: ??
            ??
    """
    # the frames immediately preceeding stim
    frame_idx = utils.closest_frame_before(clock, start)
    trial_frames = np.arange(frame_idx-pre_frames, frame_idx+post_frames)

    # is the trial outside of the frame clock
    is_beyond_clock = np.max(trial_frames) >= len(clock) or np.min(trial_frames) < 0

    if is_beyond_clock:
        return None

    frame_to_start = (start - clock[frame_idx]) / paq_rate  # time (s) from frame to trial_start
    frame_time_diff = np.diff(clock[trial_frames]) / paq_rate  # ifi (s)

    # did the function find the correct frame
    is_not_correct_frame = clock[frame_idx+1]  < start or clock[frame_idx] > start
    # the nearest frame to trial start was not during trial
    trial_not_running = frame_to_start > 1/fs
    #
    frames_not_consecutive = np.max(frame_time_diff) > 1/(fs-1)

    if trial_not_running or frames_not_consecutive:
        return None
    return trial_frames


def build_flu_array_single(run, use_spks=False, use_comps=False, use_pupil=False, 
                           prereward=False, pre_frames=30, post_frames=80, fs=30):
    ''' Build an trial by trial fluoresence array of shape [n_cells x n_frames x n_trials].

    Parameters:
    ---------------
        run: run object
            run object containing data
        pre_frames: int, default=30
            number of frames before stim to include in trial
        post_frames: int, default=80
            number of frames after stim to include in trial
        fs: int, default=30
            frame rate of imaging

    Returns:
    ---------------
        flu_array: 3D np.array
            matrix of fluorescence data
    '''

    if use_spks:
        flu = run.spks
    elif use_comps:
        flu = run.comps
    elif use_pupil:
        flu = run.pupil
    else:
        flu = run.flu

    # the frames that were actually imaged and the time (samples) that they occured
    clock = run.paqio_frames

    if prereward:
        # Time of prereward trial starts in paq samples
        trial_start = run.prereward_aligner.A_to_B(run.pre_reward)
    else:
        # Times of main behaviour trial start in paq samples
        trial_start = run.spiral_start
        # check that the number of trial starts detected by x galvo thresholding
        # matches the number of trials reported by pycontrol
        assert len(trial_start) == len(run.trial_start)

    for i, start in enumerate(trial_start):

        trial_frames = get_trial_frames_single(clock, start, pre_frames, post_frames)

        if trial_frames is None:
            flu_trial = np.full([flu.shape[0], pre_frames+post_frames], np.nan)
        else:
            flu_trial = flu[:, trial_frames]

        if i == 0:
            flu_array = flu_trial
        else:

            flu_array = np.dstack((flu_array, flu_trial))


    return np.swapaxes(flu_array,2,1)


class Session:
    """Class containing all info and data of 1 imaging session, as saved in a run.pkl file."""
    def __init__(self, mouse, run_number, pkl_path, flu_flavour='flu', remove_nan_trials=True,
                pre_seconds=4, post_seconds=6, pre_gap_seconds=0.2, post_gap_seconds=0.6,
                verbose=1, filter_threshold=10):
        """Initialize parameters and call all Class methods (except shuffle_labels()) to construct attributes.

        Parameters
        ----------
        mouse : str
            name mouse.
        run_number : int
            run number (i.e.session id of this mouse).
        pkl_path : str
            path to pickle file.
        remove_nan_trials : bool, default=True
            Remove trials with Nan Df/f data.
        pre_seconds : float, default=4
            time pre-stimulus to load into trial.
        post_seconds : float, default=6
            time post-stimulus to load into trial.
        pre_gap_seconds : float, default=0.2
            skip these seconds before PS stimulus onset
        post_gap_seconds : float, default=0.6
            skip these seconds after PS stimulus onset
        verbose : int, default=1
            verbosiness;  1: only important info (i.e. user end); 2: all info (debug end)
        filter_threshold : int, default=10 filter neurons with mean(abs(df/f)) > filter_threshold
        """
        self.mouse = mouse
        self.run_number = run_number
        self.signature = f'{self.mouse}_R{self.run_number}' # signature to use whenever
        self.pkl_path = pkl_path
        self.name = f'Mouse {mouse}, run {run_number}'
        self.run = None
        self.pre_seconds = pre_seconds  # pre stimulation
        self.post_seconds = post_seconds  # post stimulation
        self.pre_gap_seconds = pre_gap_seconds
        self.post_gap_seconds = post_gap_seconds
        self.verbose = verbose  # 1: only important info (i.e. user end); 2: all info (debug end)
        self.shuffled_s1s2_labels_indicator = False
        self.shuffled_trial_labels_indicator = False

        assert flu_flavour == 'flu', 'Only flu implemented as flu_flavour for Session object. See SessionLite for how to incorporate other values such as raw and denoised flu.'
        self.flu_flavour = flu_flavour  # otherwise not used, but added for backwards compability with SessionLite 

        self.load_data(vverbose=self.verbose)  # load data from pkl path

        if mouse=='J048' or mouse=='RL048':
            self.frequency = 5
            self.build_time_gap_array()  # requires frequency def, construct time arrat with PS gap
            self.build_trials_multi(vverbose=self.verbose)  # create Df/f matrix (index per trial )
        else:
            self.frequency = 30
            self.build_time_gap_array()
            self.build_trials_single(vverbose=self.verbose)
        self.filter_neurons(vverbose=self.verbose, abs_threshold=filter_threshold)  #filter neurons based on mean abs values
        self.define_s1_s2()   # label s1 and s2 identity of neurons
        self.label_trials(vverbose=self.verbose)  # label trial outcomes
        self.get_targets()

        self.remove_nan_trials_inplace(vverbose=self.verbose)  # remove nan trials
        self.get_first_lick_spont()
        delattr(self.run, 'x_galvo_uncaging')   # to free memory

    def __str__(self):
        """Define name"""
        return self.name

    def __repr__(self):
        """Define representation"""
        return f'instance {self.name} of Session class'

    def load_data(self, vverbose=1):
        """Load run.pkl file that contains all data about a session."""
        if vverbose >= 1:
            print(f'Now loading mouse {self.mouse}, run {self.run_number}')
        run_path = os.path.join(self.pkl_path, self.mouse, f'run{self.run_number}.pkl')
        print(run_path)
        run_path = str(run_path)
        with open(run_path, 'rb') as f:  # load data
            r = pickle.load(f)
            self.run = r

        self.suite2p_id = np.array([stat['original_index'] for stat in self.run.stat])
        
        ## Start preprocessing:
        self.flu = self.run.flu
        self.tstart_galvo = utils.threshold_detect(self.run.x_galvo_uncaging, 0)
        self.tstart_galvo = self.run.spiral_start
        self.trial_start = self.run.trial_start
        assert len(self.trial_start) == len(self.tstart_galvo)
        self.galvo_ms = self.run.aligner.B_to_A(self.tstart_galvo)
        self.first_lick = np.array([licks[0] if len(licks) > 0 else None 
                                    for licks in self.run.spiral_licks])
        if vverbose >= 1:
            print('microcontroller trial starts occur on average {} ms from galvo trial starts'
              .format(round(np.mean(self.trial_start - self.galvo_ms), 2)))

        ## Info about PS & outcome
        ### a different number of cells were stimulated on each trial
        ### need to create a Subsets object to get this info (future code refinement will
        ### include this info directly in the run object
        self.subsets = Subsets(self.run)
        self.trial_subsets = self.subsets.trial_subsets
        self.n_stim_arr = np.unique(self.trial_subsets)
        self.outcome = self.run.outcome
        self.outcome_arr = np.unique(self.outcome)

    def define_s1_s2(self, im_size=1024):  # define border (which is hard-defined at middle of image)
        """Define S1/S2 region borders and label all neurons accordingly.

        Parameters:
        -------------------
            im_size: int, default=1024
                size of imaging window in pixels. This is required because multi-plane data is saved by transposing planes +im_size pixels.
        """
        if self.run is None:
            self.load_data()
        self.n_cells = self.run.stat.shape[0]
        assert self.n_cells == self.behaviour_trials.shape[0]
        self.av_ypix = np.zeros(self.n_cells)
        self.av_xpix = np.zeros(self.n_cells)
        self.plane_number = np.zeros(self.n_cells)
        for neuron_id in range(self.n_cells):# get mean coords of pixels per cell
            self.av_xpix[neuron_id] = np.mean(self.run.stat[neuron_id]['xpix']) % im_size  # modulo 1024 because different planes are transposed by image size
            self.av_ypix[neuron_id] = np.mean(self.run.stat[neuron_id]['ypix']) % im_size
            # JR addition to deal with single plane data that has no iplane
            try:
                self.plane_number[neuron_id] = self.run.stat[neuron_id]['iplane']
            except KeyError:
                self.plane_number[neuron_id] = 0


        # This will either be a scalar -> the x coord of a vertical line separating S1 and S2,
        # or a vector describing a non straight line, in the format [x1,y1,x2,y2]
        with open('/home/jrowland/Documents/code/Vape/s2_position.json') as json_file:
            s1s2_border_json = json.load(json_file)
    
        self.s1s2_border = s1s2_border_json[self.mouse][str(self.run_number)]
        

        if isinstance(self.s1s2_border, int):
            # Straight line
            self.s2_bool = self.av_xpix > self.s1s2_border
        else:
            # Arbitrary line
            Ax, Ay, Bx, By = self.s1s2_border
            self.s2_bool = []
            for X, Y in zip(self.av_xpix, self.av_ypix):
                position = np.sign((Bx - Ax) * (Y - Ay) - (By - Ay) * (X - Ax))
                if position == 1:
                    self.s2_bool.append(False)
                else:
                    self.s2_bool.append(True)

            self.s2_bool = np.array(self.s2_bool)
                    
        self.s1_bool = np.logical_not(self.s2_bool)

    def run_oasis(self):
        """ Build spks array using Oasis deconvolution https://github.com/j-friedrich/OASIS """

        flu = self.flu.astype('float64')

        denoised_flu = np.empty_like(flu)
        deconved = np.empty_like(flu)

        for idx, cell in enumerate(tqdm(flu)):
            c, s, b, g, lam = deconvolve(cell, penalty=0)
            denoised_flu[idx, :] = c
            deconved[idx, :] = s

        self.run.flu = deconved  # Come up with a more elegant solution to this
        self.deconved = deconved
        self.denoised_flu = denoised_flu


    def build_time_gap_array(self):
        """Filter frames around PS due to laser artefact, build relevant new time/frame arrays etc."""
        self.pre_frames = int(np.round(self.pre_seconds * self.frequency))  # convert seconds to frame
        self.post_frames = int(np.round(self.post_seconds * self.frequency))
        self.art_gap_start = self.pre_frames - int(np.round(self.pre_gap_seconds * self.frequency))  # end of pre-stim frames
        self.final_pre_gap_tp = np.arange(self.art_gap_start)[-1]
        self.art_gap_stop = self.pre_frames + int(np.round(self.post_gap_seconds * self.frequency))  # start of post-stim frames
        self.filter_ps_array = np.concatenate((np.arange(self.art_gap_start),
                                  np.arange(self.art_gap_stop, self.pre_frames + self.post_frames))) # frame array of remaining frames
        self.filter_ps_time = (self.filter_ps_array - self.art_gap_start + 1) / self.frequency  # time array for remaining frames wrt stim onset

    def build_trials_multi(self, use_spks=False, vverbose=1):
        """Construct 3D matrix of neural data (n_cells x n_trials x n_frames) for multi-plane data sets."""
        # array of fluoresence through behavioural trials (n_cells x n_trials x n_frames)
        # with e.g. the first trials spanning (galvo_ms[0] - pre_frames) : (galvo_ms[0] + post_frames)
        self.behaviour_trials = utils.build_flu_array(self.run, self.galvo_ms,
                                                      pre_frames=self.pre_frames, post_frames=self.post_frames, use_spks=use_spks)
#         self.behaviour_trials = self.behaviour_trials - np.nanmean(self.behaviour_trials, (1, 2))[:, np.newaxis, np.newaxis]
        if vverbose >= 2:
            print(f'Shape new array : {self.behaviour_trials.shape}')
        assert self.behaviour_trials.shape[1] == self.outcome.shape[0]

        self.pre_rew_trials = utils.build_flu_array(self.run, self.run.pre_reward, self.post_frames,
                                                     self.pre_frames, is_prereward=True, use_spks=use_spks)

        nan_trials = np.any(np.isnan(self.pre_rew_trials), axis=(0,2))
        self.pre_rew_trials = self.pre_rew_trials[:, ~nan_trials, :]

        assert np.sum(np.isnan(self.pre_rew_trials)) == 0

        if vverbose >= 2:
            print(self.behaviour_trials.shape, self.pre_rew_trials.shape)

    def build_trials_single(self, use_spks=False, vverbose=1):
        """Construct 3D matrix of neural data (n_cells x n_trials x n_frames) for single-plane data sets."""
        # array of fluoresence through behavioural trials (n_cells x n_trials x n_frames)
        # with e.g. the first trials spanning (galvo_ms[0] - pre_frames) : (galvo_ms[0] + post_frames)
        self.behaviour_trials = build_flu_array_single(self.run, use_spks=use_spks, prereward=False,
                                                       pre_frames=self.pre_frames,
                                                       post_frames=self.post_frames, fs=30)
        if vverbose >= 2:
            print(f'Shape new array : {self.behaviour_trials.shape}')
        assert self.behaviour_trials.shape[1] == self.outcome.shape[0],\
               '{} {}'.format(self.behaviour_trials.shape[1], self.outcome.shape[0])

        self.pre_rew_trials = build_flu_array_single(self.run, use_spks=use_spks, prereward=True,
                                                     pre_frames=self.pre_frames,
                                                     post_frames=self.post_frames, fs=30)

        nan_trials = np.any(np.isnan(self.pre_rew_trials), axis=(0,2))
        self.pre_rew_trials = self.pre_rew_trials[:, ~nan_trials, :]

    def filter_neurons(self, vverbose=1, abs_threshold=10, debug=False):
        """Filter neurons with surreal stats

        Parameters:
        -----------------
            abs_threshold, float, default=10
                upper bound on mean(abs(df/f))"""
        mean_abs_df = np.max(self.run.flu, 1)
        # mean_abs_df = np.max(np.abs(self.run.flu), 1)
        #### Now uses the max rather than the mean
        self.unfiltered_n_cells = self.run.flu.shape[0]
        self.filtered_neurons = np.where(mean_abs_df < abs_threshold)[0]
        print(f'session {self.signature}: {len(self.filtered_neurons)} kept, {self.unfiltered_n_cells - len(self.filtered_neurons)} excluded, {self.unfiltered_n_cells} total')
        self.behaviour_trials = self.behaviour_trials[self.filtered_neurons, :, :]
        self.pre_rew_trials = self.pre_rew_trials[self.filtered_neurons, :, :]
        self.run.flu = self.run.flu[self.filtered_neurons, :]
        self.run.flu_raw = self.run.flu_raw[self.filtered_neurons, :]
        self.run.stat = self.run.stat[self.filtered_neurons]
        self.suite2p_id = self.suite2p_id[self.filtered_neurons]
        # self.is_target = self.is_target[self.filtered_neurons, :, :]

        if vverbose >= 1:
            if len(self.filtered_neurons) < self.unfiltered_n_cells:
                print(f'{self.unfiltered_n_cells - len(self.filtered_neurons)} / {self.unfiltered_n_cells} cells filtered')
            else:
                print('No neurons filtered')

    def find_unrewarded_hits(self):
        """Find unrewarded hit trials that are defined as
        (registered as miss) & (lick before 1000ms) & (not an autorewarded trials )"""
        self.spiral_lick = self.run.spiral_licks  # [self.run.spiral_licks[x] for x in self.nonnan_trials]
        lick_trials = np.zeros(len(self.spiral_lick))
        for x in range(len(self.spiral_lick)):
            if len(self.spiral_lick[x] > 0):  # if licks present
                lick_trials[x] = (self.spiral_lick[x] < 1000)[0]  # True if first lick < 1000ms
            else:
                lick_trials[x] = False  # if no licks, always False
        mismatch = self.decision - lick_trials.astype('int')
        assert len(mismatch) == len(self.autorewarded)
        ind_unrew_hits = np.where(np.logical_and(mismatch == -1, self.autorewarded == False))[0]
        self.unrewarded_hits = np.zeros_like(self.decision, dtype='bool')
        self.unrewarded_hits[ind_unrew_hits] = True

    def label_trials(self, vverbose=1):
        """Construct the trial labels (PS & lick), and occurence table."""
        self.decision = np.logical_or(self.outcome == 'hit', self.outcome == 'fp').astype('int')
        self.photostim = np.ones_like(self.trial_subsets)  # ones = 5-50
        self.photostim[self.trial_subsets == 0] = 0
        self.photostim[self.trial_subsets == 150] = 2
        self.photostim_occ = {x: np.sum(self.photostim == x) for x in list(np.unique(self.photostim))}
        if vverbose >= 1:
            print(f'photo stim occurences: {self.photostim_occ}')
        self.autorewarded = np.array(rf.autoreward(self.run))  # array of bools whether an autoreward (after 3 consecutive misses) has occurred
        self.find_unrewarded_hits()  # adds unrewarded hits, requires self.decision to be defined
        # TODO: fill in arm and urh in outcome 
        
        assert self.photostim.shape == self.decision.shape
        self.n_unique_stims = len(np.unique(self.photostim))
        self.n_neurons = self.behaviour_trials.shape[0]
        self.n_times = self.behaviour_trials.shape[2]
        self.n_trials = self.behaviour_trials.shape[1]
        self.n_unique_dec = len(np.unique(self.decision))
        self.occ_table = np.zeros((self.n_unique_stims, 2))  # stim x dec
        for dec in range(self.n_unique_dec):
            for stim in range(self.n_unique_stims):
                self.occ_table[stim, dec] = np.sum(np.logical_and(self.decision == dec, self.photostim == stim))
        self.n_com_trials = np.max(self.occ_table).astype('int')
        if vverbose >= 1:
            print('Occurence table:')
            print(self.occ_table)

    def load_timescales_pkl(self):
        ''' Temporary function to get the nonnan_trials from matthias' timescales pkl '''

        timescales_pkl = 'OASIS_TAU_dffDetrended_60Pre60PostStim_sessions_liteNoSPKS3_flu.pkl'
        timescales_pkl_path = os.path.join(USER_PATHS_DICT['base_path'], timescales_pkl)

        with open(timescales_pkl_path, 'rb') as f:
            timescale_sessions = pickle.load(f)

        for ts in timescale_sessions.values():
            if ts.mouse == self.mouse and ts.run_number == self.run_number:
                self.nonnan_trials = ts.nonnan_trials


    def remove_nan_trials_inplace(self, vverbose=1):
        """Identify trials for which NaN values occur in the neural activity and remove those."""
        self.nonnan_trials = None
        # self.load_timescales_pkl()
        if self.nonnan_trials is None:  # Session is not in the timescales pkl
            self.nonnan_trials = np.unique(np.where(~np.isnan(self.behaviour_trials))[1])

        if self.mouse == 'J064' and self.run_number == 14:
            # One misaligned trial
            self.nonnan_trials = np.delete(self.nonnan_trials, 
                                           np.where(np.isin(self.nonnan_trials, 
                                           [177]))[0])

        if self.mouse == 'RL117' and self.run_number == 29:
            # Some misaligned trials
            self.nonnan_trials = np.delete(self.nonnan_trials, 
                                           np.where(np.isin(self.nonnan_trials, 
                                           [88, 219, 230, 297]))[0])

        if self.mouse == 'RL117' and self.run_number == 30:
            # The blind opened and a couple misaligned
            self.nonnan_trials = np.delete(self.nonnan_trials, 
                                           np.where(np.isin(self.nonnan_trials, 
                                           [44, 124, 192, 210]))[0])

        elif self.mouse == 'RL116' and self.run_number == 32:
            # Something weird happened on this trial causing every cell to dip
            # very negative. Possible blind opening or something
            self.nonnan_trials = np.delete(self.nonnan_trials, 
                                           np.where(self.nonnan_trials==286)[0])

        elif self.mouse == 'RL116' and self.run_number == 33:
            # Alignment weird
            self.nonnan_trials = np.delete(self.nonnan_trials, 
                                           np.where(self.nonnan_trials==14)[0])

        elif self.mouse == 'RL123' and self.run_number == 22:
            # Alignment weird
            self.nonnan_trials = np.delete(self.nonnan_trials, 
                                           np.where(np.isin(self.nonnan_trials, 
                                           [51, 105, 119]))[0])


        self.behaviour_trials = self.behaviour_trials[:, self.nonnan_trials, :]
        self.photostim = self.photostim[self.nonnan_trials]
        self.decision = self.decision[self.nonnan_trials]
        self.trial_subsets = self.trial_subsets[self.nonnan_trials]
        self.outcome = self.outcome[self.nonnan_trials]
        self.autorewarded = self.autorewarded[self.nonnan_trials]
        self.unrewarded_hits = self.unrewarded_hits[self.nonnan_trials]
        self.n_trials = len(self.nonnan_trials)
        self.is_target = self.is_target[:, self.nonnan_trials, :]
        self.first_lick = self.first_lick[self.nonnan_trials]

        if vverbose >= 1:
            print(f'{len(self.nonnan_trials)} / {self.behaviour_trials.shape[1]} non nan trials identified')
            print(f'Numbers of PS cells: {np.unique(self.trial_subsets)}')  # exact amount of PS neurons
        if vverbose >= 2:
            print(f'Time array: {self.filter_ps_array}')  # time points outside of laser artefact

    def shuffle_trial_labels(self):
        """Shuffle all trial labels (in place)"""
        n_trials = len(self.photostim)
        random_inds = np.random.choice(a=n_trials, size=n_trials, replace=False)
        self.photostim = self.photostim[random_inds]
        self.decision = self.decision[random_inds]
        self.trial_subsets = self.trial_subsets[random_inds]
        self.outcome = self.outcome[random_inds]
        self.autorewarded = self.autorewarded[random_inds]
        self.unrewarded_hits = self.unrewarded_hits[random_inds]
        self.shuffled_trial_labels_indicator = True

    def shuffle_s1s2_labels(self):
        """Shuffle s1/s2 labels for all neurons"""
        n_s1 = np.sum(self.s1_bool)
        n_cells = len(self.s1_bool)
        random_inds = np.random.choice(a=n_cells, size=n_s1, replace=False)
        self.s1_bool = np.zeros(n_cells, dtype='bool')
        self.s1_bool[random_inds] = True
        self.s2_bool = np.logical_not(self.s1_bool)
        self.shuffled_s1s2_labels_indicator = True

    def get_targets(self):
        
        gt = rf.GetTargets(self.run)
        self.is_target = gt.is_target
        # print(self.is_target.shape)
        # print(self.filtered_neurons)
        # self.is_target[self.filtered_neurons, :]
        self.is_target = np.repeat(self.is_target[:, :, np.newaxis], self.n_times, axis=2)
        # Was a cell targeted on any trial?
        ever_targeted = np.any(self.is_target, axis=(1,2))
        # Check that all targets are in s1
        # print('WARNING S1 TARGET CHECKER DISABLED')
        # n = 0
        for target, s1 in zip(ever_targeted, self.s1_bool):
            if target:
                assert s1
                # if not s1:
                    # print("S1 CHECK FAILED")
                    # print(self.run.stat[n]['original_index'])
            # n += 1

    def get_first_lick_spont(self, lick_time_array=None, reward_delivery_array=None, 
                             verbose=0, store_spont_licks=True):
        if lick_time_array is None:
            lick_time_array = self.run.pre_licks 
        if reward_delivery_array is None:
            reward_delivery_array = self.run.pre_reward

        lick_time_array = copy.deepcopy(lick_time_array)
        first_lick_array = np.zeros_like(reward_delivery_array, dtype=np.float)

        for i_rew, rew_time in enumerate(reward_delivery_array):
            diff_lick_time = lick_time_array - rew_time 

            ## while loop to avoid picking negative values
            first_lick_relative = -1  # start value
            first_lick_indices = np.argsort(np.abs(diff_lick_time))
            i_ind = -1
            while first_lick_relative <= 0:
                i_ind += 1
                first_lick_index = first_lick_indices[i_ind]
                first_lick = lick_time_array[first_lick_index]
                if np.isin(first_lick, reward_delivery_array):  # something fishy here
                    continue
                first_lick_relative = first_lick - rew_time
            if verbose > 1:
                print([lick_time_array[x] for x in first_lick_indices[:5]], i_ind, rew_time)
            if verbose > 0:
                print(f'Reward at {rew_time}, first lick at {first_lick} or {first_lick_relative}')
            first_lick_array[i_rew] = first_lick_relative

            ## exception clausules for no lick:
            if i_rew == len(reward_delivery_array) - 1:  # last reward 
                if first_lick < rew_time: # no licks apparently after reward
                    assert False, 'double check - no licks after final reward time'
                    ## I'll just put an assert here - don't expect this to ever occur so might be something wrong with data if it would happen
            else:
                if first_lick > reward_delivery_array[i_rew + 1]:  # if first lick actually occurs after next reward
                    print('double check - no licks associated with this reward delivery')
                    first_lick_array[i_rew] = np.nan  # no lick => nan (= None in np arrays)
                    ## also don't expect this to happen 
        self.first_lick_spont = first_lick_array
        
        if store_spont_licks:
            self.pre_licks = self.run.pre_licks 
            self.pre_reward = self.run.pre_reward

        return first_lick_array

        
class SessionLite(Session):
    ''' Does the same job as Session, using inheritence, but removes large-size objects run and flu for memory effiency '''

    def __init__(self, mouse, run_number, pkl_path, flu_flavour, remove_nan_trials=True,
                pre_seconds=4, post_seconds=6, pre_gap_seconds=0.2, post_gap_seconds=0.6,
                verbose=1, filter_threshold=10):

        self.mouse = mouse
        self.run_number = run_number
        self.flu_flavour = flu_flavour
        self.pkl_path = pkl_path
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.pre_gap_seconds = pre_gap_seconds
        self.post_gap_seconds = post_gap_seconds
        self.verbose = verbose
        self.filter_threshold = filter_threshold
        self.name = f'Mouse {mouse}, run {run_number}'

        self.load_data()

        if self.flu_flavour == 'flu':
            use_spks = False
        elif self.flu_flavour == 'spks':
            use_spks = True
        elif self.flu_flavour == 'denoised_flu':
            use_spks = False
            self.run.dff = copy.deepcopy(self.run.flu)
            assert self.run.flu.shape == self.run.denoised_flu.shape
            # Inelegant reassignment but saves reworking downstream functions
            self.run.flu = self.run.denoised_flu
        elif self.flu_flavour == 'flu_raw':
            use_spks = False
            self.run.dff = copy.deepcopy(self.run.flu)
            assert self.run.flu.shape == self.run.flu_raw.shape
            self.run.flu = self.run.flu_raw
        else:
            raise ValueError('self.flu_flavour not recognised')

        if mouse=='J048' or mouse=='RL048':
            self.frequency = 5
            self.build_time_gap_array()  # requires frequency def, construct time arrat with PS gap
            # create Df/f matrix (index per trial )
            self.flu = self.build_trials_multi(vverbose=self.verbose, use_spks=use_spks)
        else:
            self.frequency = 30
            self.build_time_gap_array()  # requires frequency def, construct time arrat with PS gap
            self.flu = self.build_trials_single(vverbose=self.verbose, use_spks=use_spks)

        self.label_trials(vverbose=self.verbose)  # label trial outcomes
        # Filter neurons based on mean abs values
        self.filter_neurons(vverbose=self.verbose, abs_threshold=filter_threshold)  
        self.define_s1_s2()   # label s1 and s2 identity of neurons
        self.get_targets()
        self.remove_nan_trials_inplace(vverbose=self.verbose)  # remove nan trials
        self.get_first_lick_spont()
        self.clean_obj()

    # def filter_neurons(self, vverbose=1, abs_threshold_df=10, abs_threshold_spks=1):
        # """Filter neurons with surreal stats
           # Overwritten here by subclass to remove cells with 'surreal' flu and/or spks

        # Parameters:
        # -----------------
            # abs_threshold, float, default=10
                # upper bound on mean(abs(df/f)) or mean(spks)"""

        # # run.flu is reassigned above, so make sure the same neurons are filtered
        # # regardless of flu_flavour
        # if self.flu_flavour == 'denoised_flu' or self.flu_flavour == 'flu_raw':
            # mean_abs_df = np.nanmean(np.abs(self.run.dff), 1)
        # else:
            # mean_abs_df = np.nanmean(np.abs(self.run.flu), 1)

        # mean_abs_spks = np.nanmean(np.abs(self.run.spks), 1)
        # self.unfiltered_n_cells = self.run.flu.shape[0]
        # self.filtered_neurons = np.where((mean_abs_df < abs_threshold_df) &
                                         # (mean_abs_spks < abs_threshold_spks))[0]
        # self.behaviour_trials = self.behaviour_trials[self.filtered_neurons, :, :]
        # self.pre_rew_trials = self.pre_rew_trials[self.filtered_neurons, :, :]
        # self.run.flu = self.run.flu[self.filtered_neurons, :]
        # self.run.flu_raw = self.run.flu_raw[self.filtered_neurons, :]
        # self.run.stat = self.run.stat[self.filtered_neurons]
        # if vverbose >= 1:
            # if len(self.filtered_neurons < self.unfiltered_n_cells):
                # print(f'{self.unfiltered_n_cells - len(self.filtered_neurons)} / {self.unfiltered_n_cells} cells filtered')
            # else:
                # print('No neurons filtered')

    def clean_obj(self):

        attrs_remove = ['run', 'flu']

        for attr in attrs_remove:
            delattr(self, attr)


def only_numerics(seq):
    seq_type= type(seq)
    return seq_type().join(filter(seq_type.isdigit, seq))

def load_files(save_dict, data_dict, folder_path, flu_flavour,
               pre_seconds=4, post_seconds=6, session_type='lite'):
    total_ds = 0
    debug = False
    assert session_type in ['full', 'lite']
    for mouse in data_dict.keys():
        
        if mouse in ['J048', 'RL048']:  # Drop the 5Hz data for now
            continue

        if mouse != 'J065' and debug:
            continue
        for run_number in data_dict[mouse]:
            if run_number != 10 and debug:
                continue

            try:
                if session_type == 'full':
                    session = Session(mouse=mouse, run_number=run_number, pkl_path=folder_path, 
                                        flu_flavour=flu_flavour, pre_gap_seconds=0,
                                        post_gap_seconds=0, pre_seconds=pre_seconds, 
                                        post_seconds=post_seconds, 
                                        filter_threshold=10)

                elif session_type == 'lite':
                    session = SessionLite(mouse=mouse, run_number=run_number, pkl_path=folder_path, 
                                        flu_flavour=flu_flavour, pre_gap_seconds=0,
                                        post_gap_seconds=0, pre_seconds=pre_seconds, 
                                        post_seconds=post_seconds, 
                                        filter_threshold=10)

                save_dict[total_ds] = session
                total_ds += 1
                print(f'succesfully loaded mouse {mouse}, run {run_number}')
            except AttributeError as e:
                print(f'ERROR {e}')
                if debug: raise
    return save_dict, total_ds

if __name__ == '__main__':

    session_type = input('Enter session type: [full, lite]. Full will include original run and flu files and therefore have larger memory footprint.\n\n')
    assert session_type in ['full', 'lite'], f'session type {session_type} not recognised'

    if session_type == 'full':
        flu_flavour = 'flu'
    elif session_type == 'lite':    
        flu_flavour = input('Enter a flu_flavour, valid entries are: [dff, raw, denoised, spks]\n\n')

        if flu_flavour == 'dff':
            flu_flavour = 'flu'
        elif flu_flavour == 'denoised':
            flu_flavour = 'denoised_flu'
        elif flu_flavour == 'raw':
            flu_flavour = 'flu_raw'
        elif flu_flavour == 'spks':
            flu_flavour = 'spks'
        else:
            print('Invalid flu_flavour')
            time.sleep(2)
            raise ValueError

    save_option = input('Save locally (base path) or on qnap? [local, qnap]\n\n')
    assert save_option in ['local', 'qnap']

    pkl_path = USER_PATHS_DICT['pkl_path']  #'/home/jrowland/Documents/code/Vape/run_pkls'

    if not os.path.exists(pkl_path):
        raise FileNotFoundError('pkl_path directory not found, did you update data_path.json?')

    ## Load data
    sessions = {}

    all_mice = [x for x in os.listdir(pkl_path) if x[-4:] != '.pkl']

    # run_dict = {m: list(np.unique([int(only_numerics(x))
               # for x in os.listdir(pkl_path + f'/{m}')]))
               # for m in all_mice}

    run_dict =  {
    
    'J064': [10,11,14],
    'RL070': [28,29],
    'RL117': [26, 29,30],
    'RL123': [22],
    'RL116': [32,33],
    
    }

    if 'J065' in run_dict.keys() and 14 in run_dict['J065']:
        run_dict['J065'].remove(14)

    sessions, total_ds = load_files(save_dict=sessions, data_dict=run_dict,
                                    folder_path=pkl_path, flu_flavour=flu_flavour,
                                    pre_seconds=8, post_seconds=6, 
                                    session_type=session_type)

    dt = datetime.datetime.now()
    timestamp = str(dt.date())# + '-' + str(dt.hour).zfill(2) + str(dt.minute).zfill(2)      

    if save_option == 'local':
        dir_path = USER_PATHS_DICT['base_path']
    elif save_option == 'qnap':
        # dir_path = USER_PATHS_DICT['pkl_path']  # no writing permissions to qnap_Root
        dir_path = "/mnt/qnap_share/" 
    save_path = os.path.expanduser(f'{dir_path}/sessions_{session_type}_{flu_flavour}_{timestamp}.pkl')
    # dd.io.save(save_path, sessions)
    with open(save_path, 'wb') as f:
        pickle.dump(sessions, f)

