## general imports (also for subsequent analysis notebooks)
import sys
import os
path_to_vape = os.path.expanduser('~/repos/Vape')
sys.path.append(path_to_vape)
sys.path.append(os.path.join(path_to_vape, 'jupyter'))
sys.path.append(os.path.join(path_to_vape, 'utils'))
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
# import utils_funcs as utils
# import run_functions as rf
# from subsets_analysis import Subsets
import pickle
import sklearn.decomposition
from cycler import cycler
import pandas as pd
import math, cmath
from tqdm import tqdm
import scipy.stats, scipy.spatial
from Session import Session  # class that holds all data per session
plt.rcParams['axes.prop_cycle'] = cycler(color=sns.color_palette('colorblind'))

def save_figure(name, base_path='/home/jrowland/mnt/qnap/Figures/bois'):
    plt.rcParams['pdf.fonttype'] = 42
    plt.savefig(os.path.join(base_path, f'{name}.pdf'), 
                bbox_inches='tight', transparent=True)

def beh_metric(sessions, metric='accuracy',
               stim_array=[0, 5, 10, 20, 30, 40, 50]):
    """Compute metric to quantify behavioural performance for sessions.

    Parameters
    ----------
    sessions : dict of Sessions
        all sessions.
    metric : str, default='accuracy'
        what metric to compute. possibilities; 'accuracy' and 'sensitivity'.
    stim_array : list, default=[0, 5, 10, 20, 30, 40, 50]
        list of number of cells PS.

    Returns
    -------
    acc: np.array of size (n_sessions x n_stims)
        Array with metric per stim.

    """
    acc = np.zeros((len(sessions), len(stim_array)))
    for i_session, session in sessions.items():
        for i_stim, stim in enumerate(stim_array):
            trial_inds = np.where(session.trial_subsets == stim)[0]
#             if len(trial_inds) == 0:  # if no trials have this stimulus
#                 continue
            tp = np.sum(session.outcome[trial_inds] == 'hit')
            fp = np.sum(session.outcome[trial_inds] == 'fp')
            tn = np.sum(session.outcome[trial_inds] == 'cr')
            fn = np.sum(session.outcome[trial_inds] == 'miss')
            too_early = np.sum(session.outcome[trial_inds] == 'too_')
            assert (tp + fp + tn + fn + too_early) == len(session.outcome[trial_inds])
            if metric == 'accuracy':
                acc[i_session, i_stim] = (tp + tn) / (tp + fp + tn + fn)
            elif metric == 'sensitivity':
                acc[i_session, i_stim] = tp.copy() / (tp.copy() + fp.copy())
    return acc

def fun_return_2d(data):  # possibly add fancy stuff
    """Function that determines how multiple time points are handled in train_test_all_sessions().

    Parameters
    ----------
    data : 3D np array, last dim = Time
        neural data.

    Returns
    -------
    2D np.array
        where Time is averaged out.

    """
    return np.mean(data, 2)

def angle_vecs(v1, v2):
    """Compute angle between two vectors with cosine similarity.

    Parameters
    ----------
    v1 : np.array
        vector 1.
    v2 : np.array
        vector 2.

    Returns
    -------
    deg: float
        angle in degrees .

    """
    assert v1.shape == v2.shape
    v1, v2 = np.squeeze(v1), np.squeeze(v2)
    tmp = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    rad = np.arccos(tmp)
    deg = rad * 360 / (2 * np.pi)
    return deg

def mean_angle(deg):
    """Average angles (take periodicity into account).

    Parameters
    ----------
    deg : np.array of floats
        angles

    Returns
    -------
    mean
        mean of angles .

    """
    return math.degrees(cmath.phase(sum([cmath.rect(1, math.radians(d)) for d in deg])/len(deg)))

def create_dict_pred(nl, train_proj, lt):
    """Create dictionaries to save decoders predictions (used in train_test_all_sessions()).

    Parameters
    ----------
    nl : list
        name list of items to save.
    train_proj : bool
        whether to create dictionary keys for projected data.
    lt : list
        list of decoder names (e.g. ['stim', 'dec']).

    Returns
    -------
    dict_predictions_train
        dictionary for training data
    dict_predictions_test
        dictionary for test data

    """
    dict_predictions_test = {x + '_test': np.array([]) for x in nl}  # make dicts to save
    dict_predictions_train = {x + '_train': np.array([]) for x in nl}
    if train_proj:
        for x in lt:
            dict_predictions_train[f'pred_{x}_train_proj'] = np.array([])
            dict_predictions_test[f'pred_{x}_test_proj'] = np.array([])
    if len(lt) == 2:
        dict_predictions_train['angle_decoders'] = np.array([])
    return dict_predictions_train, dict_predictions_test

def train_test_all_sessions(sessions, trial_times_use=None, verbose=2, list_test = ['dec', 'stim'],
                            hitmiss_only=False, hitspont_only=False, include_150 = False, 
                            return_decoder_weights=False, spont_used_for_training=True,
                            n_split = 4, include_autoreward=False, include_unrewardedhit=False, 
                            neurons_selection='all', include_too_early=False,
                            C_value=0.2, reg_type='l2', train_projected=False, proj_dir='different'):
    """Major function that trains the decoders. It trains the decoders specified in
    list_test, for the time trial_times_use, for all sessions in sessions. The trials
    are folded n_split times (stratified over session.outcome), new decoders
    are trained for each fold and the test results  for each fold are concatenated.
    It returns pd.Dataframes with predictions, and optionally the decoder weights (if return_decoder_weights is True)

    Parameters
    ----------
    sessions : dict
        dictionary of sessions
    trial_times_use : np.array
        np.array of time points to use. len(trial_times_use) > 1, its elements are averaged with fun_return_2d()
    verbose : int, default=2
        verbosiness
    list_test : list, default=['dec' ,'stim']
        list of decoder names
    hitmiss_only : bool, default=False
        if true, only evaluate hit miss trials.
    include_150 : bool, default=False
        if true, include n_PS=150 trials
    return_decoder_weights : bool, default=False
        if True, also return decoder weights
    n_split : int, default=4
        number of data Folds
    include_autoreward : bool, default=True
        if True, include autoreward trials
    neurons_selection : str:, default='all'
        which neurons to include. possibilities: 'all', 's1', 's2'
    C_value : float, default=0.2
        regularisation value (if reg_type is an adeqaute regulariser type for sklearn.linear_model.LogisticRegression)
    reg_type : str, default='l2'
        regulariser type, possibilities: 'l2', 'l1', 'none' (and elastic net I think)
    train_projected : bool, default=False
        if True, also evaluate decoders on projected data
    proj_dir : str. default'different
        if train_projected is True, specifies on which axis to project. possibilities: 'different' or 'same'

    Returns
    -------
    df_prediction_train: pd.DataFrame
        train data predictions
    df_prediction_test: pd.DataFrame
        test data predictions
    if return_decoder_weights:
        dec_weights: dict
            weights of all decoders

    """
    assert (hitmiss_only and hitspont_only) is False
    if train_projected:
        print("TRAINING Projected")
    if hitmiss_only:
        if verbose >= 1:
            print('Using hit/miss trials only.')
        if 'stim' in list_test:
            list_test.remove('stim')  # no point in estimating stim, because only PS
    if hitspont_only:
        spont_used_for_training = True
        if verbose >= 1:
            print('Using hit/spont trials only.')
        if 'dec' in list_test:
            list_test.remove('dec')  # no point in estimating dec, because only PS
    
    name_list = ['autorewarded_miss', 'unrewarded_hit', 'outcome']  # names of details to save - whether autorewrd trial or not
    for nn in list_test:
        name_list.append('pred_' + nn)  # prediction
    for nn in ['dec', 'stim', 'reward']:
        name_list.append('true_' + nn)  # ground truth

    mouse_list = np.unique([ss.mouse for _, ss in sessions.items()])
    df_prediction_train, df_prediction_test = dict(), dict()
    if verbose >= 2:
        print(mouse_list)
    if return_decoder_weights:
        dec_weights = {xx: {} for xx in list_test}
    for mouse in mouse_list:
        angle_decoders = np.zeros((len(sessions), n_split))
        dict_predictions_train, dict_predictions_test = create_dict_pred(nl=name_list, train_proj=train_projected, lt=list_test)
        dict_predictions_test['used_for_training'] = np.array([])
        for i_session, session in sessions.items():  # loop through sessions/runs and concatenate results (in dicts)
            if session.mouse == mouse:  # only evaluate current mouse
                if verbose >= 1:
                    print(f'Mouse {mouse}, Starting loop {i_session + 1}/{len(sessions)}')
                if trial_times_use is None:
                    trial_frames_use = session.filter_ps_array[(session.final_pre_gap_tp + 1):(session.final_pre_gap_tp + 6)]
                    print('WARNING: trial_times undefined so hard-coding them (to 5 post-stim frames)')
                else:
                    trial_frames_use = []
                    for tt in trial_times_use:
                        trial_frames_use.append(session.filter_ps_array[np.where(session.filter_ps_time == tt)[0][0]])  # this will throw an error if tt not in filter_ps_time
                    trial_frames_use = np.array(trial_frames_use)
                    assert len(trial_times_use) == len(trial_frames_use)
                    if verbose >= 2:
                        print(trial_times_use, trial_frames_use)

                ## Set neuron inds
                if neurons_selection == 'all':
                    neurons_include = np.arange(session.behaviour_trials.shape[0])
                elif neurons_selection == 's1':
                    neurons_include = np.where(session.s1_bool)[0]
                elif neurons_selection == 's2':
                    neurons_include = np.where(session.s2_bool)[0]
                if verbose >= 2:
                    print(f'n neurons: {len(neurons_include)}/{session.n_cells}, {neurons_selection}')

                ## Set trial inds
                ## trial_inds: used for training & testing
                ## eval_only_inds: only used for testing (Auto Rew Miss; Un Rew Hit)
                if include_150 is False:
                    trial_inds = np.where(session.photostim < 2)[0]
                else:
                    trial_inds = np.arange(len(session.photostim))
                
                if hitmiss_only:
                    hitmiss_trials = np.where(np.logical_or(session.outcome == 'hit', session.outcome == 'miss'))[0]
                    if verbose == 2:
                        print(f'Size hm {hitmiss_trials.size}, trial inds {trial_inds.size}')
                    trial_inds = np.intersect1d(trial_inds, hitmiss_trials)
                    assert False, 'Hit miss only selected - but not implemented for eval_only_inds'
                elif hitspont_only:
                    hit_trials = np.where(session.outcome == 'hit')[0]
                    trial_inds = np.intersect1d(trial_inds, hit_trials)

                if include_autoreward is False:
                    ar_exclude = np.where(session.autorewarded == False)[0]
                    if verbose == 2:
                        print(f'{np.sum(session.autorewarded)} autorewarded trials found and excluded')
                    trial_inds = np.intersect1d(trial_inds, ar_exclude)
                else:
                    print('WARNING: ARM not excluded!')

                if include_unrewardedhit is False:
                    uhr_excluded = np.where(session.unrewarded_hits == False)[0]
                    if verbose == 2:
                        print(f'{np.sum(session.unrewarded_hits)} unrewarded_hits found and excluded')
                    trial_inds = np.intersect1d(trial_inds, uhr_excluded)
                else:
                    print('WARNING: URH not excluded!')

                if include_too_early is False:
                    too_early_excl = np.where(session.outcome != 'too_')[0]
                    trial_inds = np.intersect1d(trial_inds, too_early_excl)
                else:
                    print('WARNING: too early not excluded!')

                ## set evaluation only indices 
                eval_only_inds = np.concatenate((np.where(session.autorewarded == True)[0], 
                                                np.where(session.unrewarded_hits == True)[0]))
                eval_only_labels = ['arm'] * np.sum(session.autorewarded) + ['urh'] * np.sum(session.unrewarded_hits)
                assert len(eval_only_inds) == np.sum(session.autorewarded) + np.sum(session.unrewarded_hits)
                if hitmiss_only:
                    ## to implement 
                    assert False
                elif hitspont_only:
                    for tt in ['miss', 'fp', 'cr']:
                        eval_only_inds = np.concatenate((eval_only_inds, 
                                                         np.where(session.outcome == tt)[0]))
                        eval_only_labels = eval_only_labels + [tt] * len(np.where(session.outcome == tt)[0])
                    trial_outcomes = session.outcome[trial_inds]
                    assert len(np.unique(trial_outcomes)) == 1 and trial_outcomes[0] == 'hit'
                else:
                    trial_outcomes = session.outcome[trial_inds]
                    
                ## Prepare data with selections
                ## Filter neurons
                data_use = session.behaviour_trials[neurons_include, :, :]
                data_eval = session.behaviour_trials[neurons_include, :, :]
                data_spont  = session.pre_rew_trials[neurons_include, :, :]
                ## Select time frame(s)
                data_use = data_use[:, :, trial_frames_use]
                data_eval = data_eval[:, :, trial_frames_use]
                data_spont = data_spont[:, :, trial_frames_use]  # use all trials
                ## Select trials
                data_use = data_use[:, trial_inds, :]
                assert len(eval_only_inds) > 0, f'ERROR: {session} has no eval-only trials, which has not been really taken care of here'
                data_eval = data_eval[:, eval_only_inds, :]
                assert data_spont.ndim == 3
                n_spont_trials = data_spont.shape[1]
                if n_spont_trials == 0:
                    print('NO SPONT TRIALS in ', session)
                if spont_used_for_training:
                    data_use = np.hstack((data_use, data_spont))
                    trial_outcomes = np.concatenate((trial_outcomes, ['spont'] * n_spont_trials))
                    stim_trials = np.concatenate((session.photostim[trial_inds], np.zeros(n_spont_trials)))
                    dec_trials = np.concatenate((session.decision[trial_inds], np.ones(n_spont_trials)))

                    detailed_ps_labels = np.concatenate((session.trial_subsets[trial_inds].astype('int'), np.zeros(n_spont_trials)))
                    rewarded_trials = np.concatenate((np.logical_or(session.outcome[trial_inds] == 'hit', session.outcome[trial_inds] == 'too_'), np.ones(n_spont_trials)))
                    if hitspont_only:
                        assert len(rewarded_trials) == np.sum(rewarded_trials)
                    autorewarded = np.concatenate((session.autorewarded[trial_inds], np.zeros(n_spont_trials, dtype='bool')))
                    rewarded_trials[autorewarded] = True
                    unrewarded_hits = np.concatenate((session.unrewarded_hits[trial_inds], np.zeros(n_spont_trials, dtype='bool')))
                    rewarded_trials[unrewarded_hits] = False
                else:
                    stim_trials = session.photostim[trial_inds]
                    dec_trials = session.decision[trial_inds]
                    detailed_ps_labels = session.trial_subsets[trial_inds].astype('int')
                    rewarded_trials = np.logical_or(session.outcome[trial_inds] == 'hit', session.outcome[trial_inds] == 'too_')
                    autorewarded = session.autorewarded[trial_inds]
                    rewarded_trials[autorewarded] = True
                    unrewarded_hits = session.unrewarded_hits[trial_inds]
                    rewarded_trials[unrewarded_hits] = False
                assert len(trial_outcomes) == data_use.shape[1]
                ## Squeeze time frames
                data_use = fun_return_2d(data_use)
                data_eval = fun_return_2d(data_eval)
                data_spont = fun_return_2d(data_spont)
                ## Stack & fit scaler
                if spont_used_for_training:
                    data_stacked = np.hstack((data_use, data_eval))  # stack for scaling
                    assert (data_stacked[:, (len(trial_inds) + n_spont_trials):(len(trial_inds) + len(eval_only_inds) + n_spont_trials)] == data_eval).all()
                else:
                    data_stacked = np.hstack((data_use, data_eval, data_spont))  # stack for scaling
                    assert (data_stacked[:, len(trial_inds):(len(trial_inds) + len(eval_only_inds))] == data_eval).all()
                stand_scale = sklearn.preprocessing.StandardScaler().fit(data_stacked.T) # use all data to fit scaler, then scale indiviudally
                ## Scale
                data_use = stand_scale.transform(data_use.T)
                data_eval = stand_scale.transform(data_eval.T)
                data_spont = stand_scale.transform(data_spont.T)
                data_use = data_use.T
                data_eval = data_eval.T
                data_spont = data_spont.T
                sss = sklearn.model_selection.StratifiedKFold(n_splits=n_split)  # split into n_split data folds of trials
                if verbose == 2:
                    print(f'Number of licks: {np.sum(session.decision[trial_inds])}')
                    dict_outcomes = {x: np.sum(trial_outcomes == x) for x in np.unique(trial_outcomes)}
                    print(f'Possible trial outcomes: {dict_outcomes}')
                    dict_n_ps = {x: np.sum(session.trial_subsets[trial_inds] == x) for x in np.unique(session.trial_subsets[trial_inds])}
                    print(f'Possible stimulations: {dict_n_ps}')

                i_loop = 0
                if return_decoder_weights:
                    for x in list_test:
                        dec_weights[x][session.signature] = np.zeros((n_split, len(neurons_include)))

                n_trials = data_use.shape[1]
                if verbose == 2:
                    print(f'Total number of trials is {n_trials}. Number of splits is {n_split}')
                
                pred_proba_eval = {x: {} for x in range(n_split)} # dict per cv loop, average later.
                pred_proba_spont = {x: {} for x in range(n_split)}
                for train_inds, test_inds in sss.split(X=np.zeros(n_trials), y=trial_outcomes):  # loop through different train/test folds, concat results
                    train_data, test_data = data_use[:, train_inds], data_use[:, test_inds]
                    if i_loop == 0:
                        if verbose == 2:
                            print(f'Shape train data {train_data.shape}, test data {test_data.shape}')

                    ## Get labels and categories of trials
                    train_labels = {'stim': stim_trials[train_inds],
                                    'dec': dec_trials[train_inds]}
                    test_labels = {'stim': stim_trials[test_inds],
                                   'dec': dec_trials[test_inds]}
                    if verbose == 2:
                        print(f' Number of test licks {np.sum(test_labels["dec"])}')
                    assert len(train_labels['dec']) == train_data.shape[1]
                    assert len(test_labels['stim']) == test_data.shape[1]

                    ## Train logistic regression model on train data
                    dec = {}
                    for x in list_test:
                        dec[x] = sklearn.linear_model.LogisticRegression(penalty=reg_type, C=C_value, class_weight='balanced').fit(
                                        X=train_data.transpose(), y=train_labels[x])
                        if return_decoder_weights:
                            dec_weights[x][session.signature][i_loop, :] = dec[x].coef_.copy()

                    if len(list_test) == 2:
                        angle_decoders[i_session, i_loop] = angle_vecs(dec[list_test[0]].coef_, dec[list_test[1]].coef_)

                    if train_projected:  # project and re decode
                        dec_proj = {}
                        assert len(list_test) == 2  # hard coded that len==2 further on
                        for i_x, x in enumerate(list_test):
                            i_y = 1 - i_x
                            y = list_test[i_y]
                            assert x != y
                            if proj_dir == 'same':
                                enc_vector = dec[x].coef_ / np.linalg.norm(dec[x].coef_)
                            elif proj_dir == 'different':
                                enc_vector = dec[y].coef_ / np.linalg.norm(dec[y].coef_)
                            train_data_proj = enc_vector.copy() * train_data.transpose()
                            test_data_proj = enc_vector.copy() * test_data.transpose()
                            dec_proj[x] = sklearn.linear_model.LogisticRegression(penalty=reg_type, C=C_value, class_weight='balanced').fit(
                                            X=train_data_proj, y=train_labels[x])

                    ## Predict test data
                    pred_proba_train = {x: dec[x].predict_proba(X=train_data.transpose())[:, 1] for x in list_test}
                    pred_proba_test = {x: dec[x].predict_proba(X=test_data.transpose())[:, 1] for x in list_test}
                    pred_proba_eval[i_loop] = {x: dec[x].predict_proba(X=data_eval.transpose())[:, 1] for x in list_test}
                    pred_proba_spont[i_loop] = {x: dec[x].predict_proba(X=data_spont.transpose())[:, 1] for x in list_test} 
                    
                    if train_projected:
                        pred_proba_train_proj = {x: dec_proj[x].predict_proba(X=train_data_proj)[:, 1] for x in list_test}
                        pred_proba_test_proj = {x: dec_proj[x].predict_proba(X=test_data_proj)[:, 1] for x in list_test}

                    ## Save results
                    for x in list_test:
                        dict_predictions_train[f'pred_{x}_train'] = np.concatenate((dict_predictions_train[f'pred_{x}_train'], pred_proba_train[x]))
                        dict_predictions_test[f'pred_{x}_test'] = np.concatenate((dict_predictions_test[f'pred_{x}_test'], pred_proba_test[x]))
                        if train_projected:
                            dict_predictions_train[f'pred_{x}_train_proj'] = np.concatenate((dict_predictions_train[f'pred_{x}_train_proj'], pred_proba_train_proj[x]))
                            dict_predictions_test[f'pred_{x}_test_proj'] = np.concatenate((dict_predictions_test[f'pred_{x}_test_proj'], pred_proba_test_proj[x]))
                    if len(list_test) == 2:
                        dict_predictions_train['angle_decoders'] = np.concatenate((dict_predictions_train['angle_decoders'], np.zeros_like(pred_proba_train[x]) + angle_decoders[i_session, i_loop]))
                    dict_predictions_train['true_stim_train'] = np.concatenate((dict_predictions_train['true_stim_train'], detailed_ps_labels[train_inds]))
                    dict_predictions_test['true_stim_test'] = np.concatenate((dict_predictions_test['true_stim_test'], detailed_ps_labels[test_inds]))
                    dict_predictions_train['true_reward_train'] = np.concatenate((dict_predictions_train['true_reward_train'], rewarded_trials[train_inds]))
                    dict_predictions_test['true_reward_test'] = np.concatenate((dict_predictions_test['true_reward_test'], rewarded_trials[test_inds]))
                    dict_predictions_train['outcome_train'] = np.concatenate((dict_predictions_train['outcome_train'], trial_outcomes[train_inds]))
                    dict_predictions_test['outcome_test'] = np.concatenate((dict_predictions_test['outcome_test'], trial_outcomes[test_inds]))
                    dict_predictions_train['autorewarded_miss_train'] = np.concatenate((dict_predictions_train['autorewarded_miss_train'], autorewarded[train_inds]))
                    dict_predictions_test['autorewarded_miss_test'] = np.concatenate((dict_predictions_test['autorewarded_miss_test'], autorewarded[test_inds]))
                    dict_predictions_train['unrewarded_hit_train'] = np.concatenate((dict_predictions_train['unrewarded_hit_train'], unrewarded_hits[train_inds]))
                    dict_predictions_test['unrewarded_hit_test'] = np.concatenate((dict_predictions_test['unrewarded_hit_test'], unrewarded_hits[test_inds]))
                    dict_predictions_train['true_dec_train'] = np.concatenate((dict_predictions_train['true_dec_train'], train_labels['dec']))
                    dict_predictions_test['true_dec_test'] = np.concatenate((dict_predictions_test['true_dec_test'], test_labels['dec']))
                    dict_predictions_test['used_for_training'] = np.concatenate((dict_predictions_test['used_for_training'], np.ones(len(test_inds))))
                    i_loop += 1

                ## Add results of eval_only trials (average of decoder CVs):

                ## eval onlY:
                assert (np.array(list(pred_proba_eval.keys())) == np.arange(n_split)).all()
                for x in list_test:
                    mat_predictions = np.array([pred_proba_eval[nn][x] for nn in range(n_split)])                
                    assert mat_predictions.shape[0] == n_split
                    dict_predictions_test[f'pred_{x}_test'] = np.concatenate((dict_predictions_test[f'pred_{x}_test'], np.mean(mat_predictions, 0)))  
                dict_predictions_test['true_stim_test'] = np.concatenate((dict_predictions_test['true_stim_test'], session.trial_subsets[eval_only_inds].astype('int')))
                tmp_rewarded_all_trials = np.logical_or(session.outcome == 'hit', session.outcome == 'too_')
                tmp_rewarded_all_trials[session.autorewarded] = True
                tmp_rewarded_all_trials[session.unrewarded_hits] = False
                dict_predictions_test['true_reward_test'] = np.concatenate((dict_predictions_test['true_reward_test'], tmp_rewarded_all_trials[eval_only_inds]))
                dict_predictions_test['outcome_test'] = np.concatenate((dict_predictions_test['outcome_test'], eval_only_labels))
                dict_predictions_test['autorewarded_miss_test'] = np.concatenate((dict_predictions_test['autorewarded_miss_test'], session.autorewarded[eval_only_inds]))
                dict_predictions_test['unrewarded_hit_test'] = np.concatenate((dict_predictions_test['unrewarded_hit_test'], session.unrewarded_hits[eval_only_inds]))
                dict_predictions_test['true_dec_test'] = np.concatenate((dict_predictions_test['true_dec_test'], session.decision[eval_only_inds]))
                dict_predictions_test['used_for_training'] = np.concatenate((dict_predictions_test['used_for_training'], np.zeros(len(eval_only_inds))))

                ## spontaneous:
                if n_spont_trials > 0 and (spont_used_for_training is False):
                    assert (np.array(list(pred_proba_spont.keys())) == np.arange(n_split)).all()
                    for x in list_test:
                        mat_predictions = np.array([pred_proba_spont[nn][x] for nn in range(n_split)])                
                        assert mat_predictions.shape[0] == n_split, mat_predictions.shape[1] == n_spont_trials
                        dict_predictions_test[f'pred_{x}_test'] = np.concatenate((dict_predictions_test[f'pred_{x}_test'], np.mean(mat_predictions, 0)))
                    dict_predictions_test['true_stim_test'] = np.concatenate((dict_predictions_test['true_stim_test'], np.zeros(n_spont_trials)))
                    dict_predictions_test['true_reward_test'] = np.concatenate((dict_predictions_test['true_reward_test'], np.ones(n_spont_trials)))
                    dict_predictions_test['outcome_test'] = np.concatenate((dict_predictions_test['outcome_test'], np.array(['spont'] * n_spont_trials)))
                    dict_predictions_test['autorewarded_miss_test'] = np.concatenate((dict_predictions_test['autorewarded_miss_test'], np.zeros(n_spont_trials)))
                    dict_predictions_test['unrewarded_hit_test'] = np.concatenate((dict_predictions_test['unrewarded_hit_test'], np.zeros(n_spont_trials)))
                    dict_predictions_test['true_dec_test'] = np.concatenate((dict_predictions_test['true_dec_test'], np.ones(n_spont_trials)))
                    dict_predictions_test['used_for_training'] = np.concatenate((dict_predictions_test['used_for_training'], np.zeros(n_spont_trials)))

        if verbose == 2:
            print(f'length test: {len(dict_predictions_test["true_dec_test"])}')
        ## Put dictionary results into dataframes:
        df_prediction_train[mouse] = pd.DataFrame(dict_predictions_train)
        df_prediction_test[mouse] = pd.DataFrame(dict_predictions_test)

    if return_decoder_weights is False:
        return df_prediction_train, df_prediction_test, None, (data_use, trial_outcomes)
    elif return_decoder_weights:
        return df_prediction_train, df_prediction_test, dec_weights, (data_use, trial_outcomes)


## Some functions that can be used as accuracy assessment
def prob_correct(binary_truth, estimate):
    """Return probability of correct estimate, where bt = {0, 1} and est = (0, 1).

    Parameters
    ----------
    binary_truth : np.array of 1s and 0s
        Binary ground truth array.
    estimate : np.array of floats 0 < f < 1
        Predictions of numbers in binary_truth.

    Returns
    -------
    prob, np.array of floats
        Accuracy (probability of being correct for each element)

    """
    prob = (binary_truth * estimate + (1 - binary_truth) * (1 - estimate))
    return prob

def mean_accuracy(binary_truth, estimate):
    """Mean accuracy (average over all trials)

    Parameters
    ----------
    binary_truth : np.array of 1s and 0s
        Binary ground truth array.
    estimate : np.array of floats 0 < f < 1
        Predictions of numbers in binary_truth.

    Returns
    -------
    mean, float
        Mean of accuracies
    std, float
        std of accuracies

    """
    assert len(binary_truth) == len(estimate)
    pp = prob_correct(binary_truth=binary_truth, estimate=estimate)
    return np.mean(pp), np.std(pp)

def mean_accuracy_pred(binary_truth, estimate):
    """Mean accuracy with hard >0.5 threshold (average of all trials)

    Parameters
    ----------
    binary_truth : np.array of 1s and 0s
        Binary ground truth array.
    estimate : np.array of floats 0 < f < 1
        Predictions of numbers in binary_truth.

    Returns
    -------
    mean_pred, float
        mean accuracy of thresholded predictions
    0

    """
    round_est = np.round(estimate)
    return sklearn.metrics.accuracy_score(binary_truth, round_est), 0

def llh(binary_truth, estimate):
    """Log likelihood of all trials.

    Parameters
    ----------
    binary_truth : np.array of 1s and 0s
        Binary ground truth array.
    estimate : np.array of floats 0 < f < 1
        Predictions of numbers in binary_truth.

    Returns
    -------
    llh, float
        Log likelihood of accuracies
    0

    """
    assert len(binary_truth) == len(estimate)
    pp = prob_correct(binary_truth=binary_truth, estimate=estimate)
    llh = np.mean(np.log(np.clip(pp, a_min=1e-3, a_max=1)))
    return llh, 0


def score_nonbinary(model, X, y):

    ''' JR helper for Thijs juice'''

    estimate = model.predict_proba(X)[:, 1]
    acc = mean_accuracy(binary_truth=y, estimate=estimate)
    return acc[0]


def r2_acc(binary_truth, estimate):
    """R2, plainly averaged over all trials (not variance-weighted).

    Parameters
    ----------
    binary_truth : np.array of 1s and 0s
        Binary ground truth array.
    estimate : np.array of floats 0 < f < 1
        Predictions of numbers in binary_truth.

    Returns
    -------
    r2, float
        R2 score of accuracies
    0

    """
    return sklearn.metrics.r2_score(y_true=binary_truth, y_pred=estimate), 0

def separability(binary_truth, estimate):
    """Measure difference between averages P(1) and P(0).

    Parameters
    ----------
    binary_truth : np.array of 1s and 0s
        Binary ground truth array.
    estimate : np.array of floats 0 < f < 1
        Predictions of numbers in binary_truth.

    Returns
    -------
    sep, float
        Separability between average class predictions
    0

    """
    av_pred_0 = np.mean(estimate[binary_truth == 0])
    av_pred_1 = np.mean(estimate[binary_truth == 1])
    sep = av_pred_1 - av_pred_0
    return sep, 0

def min_mean_accuracy(binary_truth, estimate):
    """Minimum of averages P(1) and P(0).

    Parameters
    ----------
    binary_truth : np.array of 1s and 0s
        Binary ground truth array.
    estimate : np.array of floats 0 < f < 1
        Predictions of numbers in binary_truth.

    Returns
    -------
    min_mean, float
        class-minimum of accuracies
    0

    """
    mean_acc_true = np.mean(estimate[binary_truth == 1])
    mean_acc_false = 1 - np.mean(estimate[binary_truth == 0])
    return np.minimum(mean_acc_true, mean_acc_false), 0

def class_av_mean_accuracy(binary_truth, estimate):
    """Mean of averages P(1) and P(0).
    #TODO: should we use sample correct (n-1)/n for std calculation?

    Parameters
    ----------
    binary_truth : np.array of 1s and 0s
        Binary ground truth array.
    estimate : np.array of floats 0 < f < 1
        Predictions of numbers in binary_truth.

    Returns
    -------
    class_av_mean, float
        Average accuracy where classes are weighted equally (indep of number of elements per class)
    0

    """
    if np.sum(binary_truth == 1) > 0:
        n_true = np.sum(binary_truth == 1)
        mean_acc_true = np.mean(estimate[binary_truth == 1])
        std_acc_true = np.std(estimate[binary_truth == 1])
        bin_truth_1 = True
    else:  # no BT == 1
        bin_truth_1 = False
    if np.sum(binary_truth == 0) > 0:
        n_false = np.sum(binary_truth == 0)
        mean_acc_false = 1 - np.mean(estimate[binary_truth == 0])
        std_acc_false = np.std(estimate[binary_truth == 0])
        bin_truth_0 = True
    else:  # no BT == 0
        bin_truth_0 = False
    if bin_truth_1 and bin_truth_0:
        comp_std = np.sqrt((n_true * (std_acc_true ** 2) + n_false * (std_acc_false ** 2)) / (n_true + n_false))
        return 0.5 * (mean_acc_true + mean_acc_false), comp_std
    elif bin_truth_1 and not bin_truth_0:  # if only 1 is present, return that accuracy only
        return mean_acc_true, std_acc_true
    elif not bin_truth_1 and bin_truth_0:
        return mean_acc_false, std_acc_false

## Main function to compute accuracy of decoders per time point
def compute_accuracy_time_array(sessions, time_array, average_fun=class_av_mean_accuracy, reg_type='l2',
                                region_list=['s1', 's2'], regularizer=0.02, projected_data=False):
    """Compute accuracy of decoders for all time steps in time_array, for all sessions
    Idea is that results here are concatenated overall, not saved per mouse only but
    this needs checking #TODO

    Parameters
    ----------
    sessions : dict of Session
            data
    time_array : np.array
         array of time points to evaluate
    average_fun : function
        function that computes accuracy metric
    reg_type : str, 'l2' or 'none'
        type of regularisation
    region_list : str, default=['s1', 's2']
        list of regions to compute
    regularizer : float
        if reg_type == 'l2', this is the reg strength (C in scikit-learn)
    projected_data : bool, default=False
        if true, also compute test prediction on projected data (see train_test_all_sessions())

    Returns
    -------
    tuple
        (lick_acc,
            lick accuracy of lick decoder per mouse/session
        lick_acc_split,
            lick accuracy split by trial type
        ps_acc,
            ps accuracy
        ps_acc_split,
            ps accuracy split by lick trial type
        lick_half,
            accuracy of naive fake data
        angle_dec,
            angle between decoders
        decoder_weights)
            weights of decoders

    """
    mouse_list = np.unique([ss.mouse for _, ss in sessions.items()])
    stim_list = [0, 5, 10, 20, 30, 40, 50]  # hard coded!
    tt_list = ['hit', 'fp', 'miss', 'cr']
    dec_list = [0, 1]  # hard_coded!!
    mouse_s_list = []
    for mouse in mouse_list:
        for reg in region_list:
            mouse_s_list.append(mouse + '_' + reg)
    n_timepoints = len(time_array)
    signature_list = [session.signature for _, session in sessions.items()]

    lick_acc = {reg: np.zeros((n_timepoints, 2)) for reg in region_list} #mean, std
#     lick_acc_split = {x: {reg: np.zeros((n_timepoints, 2)) for reg in region_list} for x in stim_list}  # split per ps conditoin
    lick_acc_split = {x: {reg: np.zeros((n_timepoints, 2)) for reg in region_list} for x in tt_list}  # split per tt
    lick_half = {reg: np.zeros((n_timepoints, 2)) for reg in region_list}  # naive with P=0.5 for 2 options (lick={0, 1})
    ps_acc = {reg: np.zeros((n_timepoints, 2)) for reg in region_list}
    ps_acc_split = {x: {reg: np.zeros((n_timepoints, 2)) for reg in region_list} for x in dec_list}  # split per lick conditoin
    angle_dec = {reg: np.zeros((n_timepoints, 2)) for reg in region_list}
    decoder_weights = {'s1_stim': {session.signature: np.zeros((np.sum(session.s1_bool), len(time_array))) for _, session in sessions.items()},
                       's2_stim': {session.signature: np.zeros((np.sum(session.s2_bool), len(time_array))) for _, session in sessions.items()},
                       's1_dec': {session.signature: np.zeros((np.sum(session.s1_bool), len(time_array))) for _, session in sessions.items()},
                       's2_dec': {session.signature: np.zeros((np.sum(session.s2_bool), len(time_array))) for _, session in sessions.items()}}

    for i_tp, tp in tqdm(enumerate(time_array)):  # time array IN SECONDS

        for reg in region_list:
            df_prediction_test = {reg: {}}  # necessary for compability with violin plot df custom function
            df_prediction_train, df_prediction_test[reg][tp], dec_w, _ = train_test_all_sessions(sessions=sessions, trial_times_use=np.array([tp]),
                                                          verbose=0, hitmiss_only=False, include_150=False,
                                                          include_autoreward=False, C_value=regularizer, reg_type=reg_type,
                                                          train_projected=projected_data, return_decoder_weights=True,
                                                          neurons_selection=reg)
            for xx in ['stim', 'dec']:
                for signat in signature_list:
                    decoder_weights[f'{reg}_{xx}'][signat][:, i_tp] = np.mean(dec_w[xx][signat], 0)

            tmp_dict = make_violin_df_custom(input_dict_df=df_prediction_test,
                                           flat_normalise_ntrials=False, verbose=0)
            total_df_test = tmp_dict[tp]
            lick = total_df_test['true_dec_test'].copy()
            ps = (total_df_test['true_stim_test'] > 0).astype('int').copy()
            if projected_data is False:
                pred_lick = total_df_test['pred_dec_test'].copy()
            else:
                pred_lick = total_df_test['pred_dec_test_proj']
            lick_half[reg][i_tp, :] = average_fun(binary_truth=lick, estimate=(np.zeros_like(lick) + 0.5))  # control for P=0.5
            lick_acc[reg][i_tp, :] = average_fun(binary_truth=lick, estimate=pred_lick)

            for x, arr in lick_acc_split.items():
                arr[reg][i_tp, :] = average_fun(binary_truth=lick[np.where(total_df_test['outcome_test'] == x)[0]],
                                          estimate=pred_lick[np.where(total_df_test['outcome_test'] == x)[0]])

            if 'pred_stim_test' in total_df_test.columns:
                if projected_data is False:
                    pred_ps = total_df_test['pred_stim_test']
                else:
                    pred_ps = total_df_test['pred_stim_test_proj']
                ps_acc[reg][i_tp, :] = average_fun(binary_truth=ps, estimate=pred_ps)

                for x, arr in ps_acc_split.items():
                    arr[reg][i_tp, :] = average_fun(binary_truth=ps[lick == x],
                                              estimate=pred_ps[lick == x])
            tmp_all_angles = np.array([])
            for mouse in df_prediction_train.keys():
                tmp_all_angles = np.concatenate((tmp_all_angles, df_prediction_train[mouse]['angle_decoders']))
            angle_dec[reg][i_tp, 0] = mean_angle(tmp_all_angles)  # not sure about periodic std????

    return (lick_acc, lick_acc_split, ps_acc, ps_acc_split, lick_half, angle_dec, decoder_weights)

## Main function to compute accuracy of decoders per time point
def compute_accuracy_time_array_average_per_mouse(sessions, time_array, average_fun=class_av_mean_accuracy, reg_type='l2',
                                                  region_list=['s1', 's2'], regularizer=0.02, projected_data=False, split_fourway=False,
                                                  tt_list=['hit', 'fp', 'miss', 'cr', 'arm', 'urh', 'spont']):
    """Compute accuracy of decoders for all time steps in time_array, for all sessions (concatenated per mouse)

    Parameters
    ----------
    sessions : dict of Session
            data
    time_array : np.array
         array of time points to evaluate
    average_fun : function
        function that computes accuracy metric
    reg_type : str, 'l2' or 'none'
        type of regularisation
    region_list : str, default=['s1', 's2']
        list of regions to compute
    regularizer : float
        if reg_type == 'l2', this is the reg strength (C in scikit-learn)
    projected_data : bool, default=False
        if true, also compute test prediction on projected data (see train_test_all_sessions())

    Returns
    -------
    tuple
        (lick_acc,
            lick accuracy of lick decoder per mouse/session
        lick_acc_split,
            lick accuracy split by trial type
        ps_acc,
            ps accuracy
        ps_acc_split,
            ps accuracy split by lick trial type
        lick_half,
            accuracy of naive fake data
        angle_dec,
            angle between decoders
        decoder_weights)
            weights of decoders

    """
    mouse_list = np.unique([ss.mouse for _, ss in sessions.items()])
    stim_list = [0, 5, 10, 20, 30, 40, 50]  # hard coded!
    dec_list = [0, 1]  # hard_coded!!
    mouse_s_list = []
    for mouse in mouse_list:
        for reg in region_list:
            mouse_s_list.append(mouse + '_' + reg)
    n_timepoints = len(time_array)
    signature_list = [session.signature for _, session in sessions.items()]

    lick_acc = {mouse: np.zeros((n_timepoints, 2)) for mouse in mouse_s_list} #mean, std
#     lick_acc_split = {x: {mouse: np.zeros((n_timepoints, 2)) for mouse in mouse_s_list} for x in stim_list}  # split per ps conditoin
    lick_acc_split = {x: {mouse: np.zeros((n_timepoints, 2)) for mouse in mouse_s_list} for x in tt_list}  # split per tt
    lick_half = {mouse: np.zeros((n_timepoints, 2)) for mouse in mouse_s_list}  # naive with P=0.5 for 2 options (lick={0, 1})
    ps_acc = {mouse: np.zeros((n_timepoints, 2)) for mouse in mouse_s_list}
    if split_fourway is False:
        ps_acc_split = {x: {mouse: np.zeros((n_timepoints, 2)) for mouse in mouse_s_list} for x in dec_list}  # split per trial type
        ps_pred_split = {x: {mouse: np.zeros((n_timepoints, 2)) for mouse in mouse_s_list} for x in dec_list} 
    elif split_fourway is True:
        ps_acc_split = {x: {mouse: np.zeros((n_timepoints, 2)) for mouse in mouse_s_list} for x in tt_list}  # split per lick condition
        ps_pred_split = {x: {mouse: np.zeros((n_timepoints, 2)) for mouse in mouse_s_list} for x in tt_list} 
    angle_dec = {mouse: np.zeros(n_timepoints) for mouse in mouse_s_list}
    decoder_weights = {'s1_stim': {session.signature: np.zeros((np.sum(session.s1_bool), len(time_array))) for _, session in sessions.items()},
                       's2_stim': {session.signature: np.zeros((np.sum(session.s2_bool), len(time_array))) for _, session in sessions.items()},
                       's1_dec': {session.signature: np.zeros((np.sum(session.s1_bool), len(time_array))) for _, session in sessions.items()},
                       's2_dec': {session.signature: np.zeros((np.sum(session.s2_bool), len(time_array))) for _, session in sessions.items()}}
    for i_tp, tp in tqdm(enumerate(time_array)):  # time array IN SECONDS
        for reg in region_list:
            df_prediction_train, df_prediction_test, dec_w, _ = train_test_all_sessions(sessions=sessions, trial_times_use=np.array([tp]),
                                                          verbose=0, hitmiss_only=False, include_150=False,
                                                          include_autoreward=False, C_value=regularizer, reg_type=reg_type,
                                                          train_projected=projected_data, return_decoder_weights=True,
                                                          neurons_selection=reg)
            for xx in dec_w.keys():
                for signat in signature_list:
                    decoder_weights[f'{reg}_{xx}'][signat][:, i_tp] = np.mean(dec_w[xx][signat], 0)

            for mouse in df_prediction_train.keys():
                assert df_prediction_test[mouse][df_prediction_test[mouse]['used_for_training'] == 1]['unrewarded_hit_test'].sum() == 0
                assert df_prediction_test[mouse][df_prediction_test[mouse]['used_for_training'] == 1]['autorewarded_miss_test'].sum() == 0
            
                inds_training = np.where(df_prediction_test[mouse]['used_for_training'] == 1)[0]
                lick = df_prediction_test[mouse]['true_dec_test'].copy()
                ps = (df_prediction_test[mouse]['true_stim_test'] > 0).astype('int').copy()
                if 'pred_dec_test' in df_prediction_test[mouse].columns:
                    if projected_data is False:
                        pred_lick = df_prediction_test[mouse]['pred_dec_test'].copy()
                    else:
                        pred_lick = df_prediction_test[mouse]['pred_dec_test_proj']
                    lick_half[mouse + '_' + reg][i_tp, :] = average_fun(binary_truth=lick[inds_training], estimate=(np.zeros_like(lick[inds_training]) + 0.5))  # control for P=0.5
                    lick_acc[mouse + '_' + reg][i_tp, :] = average_fun(binary_truth=lick[inds_training], estimate=pred_lick[inds_training])
    #                 lick_acc[mouse + '_' + reg][i_tp, :] = 0
    #                 for i_lick in np.unique(lick):
    #                     lick_acc[mouse + '_' + reg][i_tp, :] += np.array(average_fun(binary_truth=lick[lick == i_lick], estimate=pred_lick[lick == i_lick])) / len(np.unique(lick))

                    for x, arr in lick_acc_split.items():
                        arr[mouse + '_' + reg][i_tp, :] = average_fun(binary_truth=lick[np.where(df_prediction_test[mouse]['outcome_test'] == x)[0]],
                                                                    estimate=pred_lick[np.where(df_prediction_test[mouse]['outcome_test'] == x)[0]])

                if 'pred_stim_test' in df_prediction_test[mouse].columns:
                    if projected_data is False:
                        pred_ps = df_prediction_test[mouse]['pred_stim_test']
                    else:
                        pred_ps = df_prediction_test[mouse]['pred_stim_test_proj']
                    ps_acc[mouse + '_' + reg][i_tp, :] = average_fun(binary_truth=ps[inds_training], estimate=pred_ps[inds_training])
#                     ps_acc[mouse + '_' + reg][i_tp, :] = 0
#                     for i_ps in np.unique(lick):
#                         ps_acc[mouse + '_' + reg][i_tp, :] += np.array(average_fun(binary_truth=ps[lick == i_ps], estimate=pred_ps[lick == i_ps])) / len(np.unique(lick))

                    for x, arr in ps_acc_split.items():
                        if split_fourway is False:  # split two ways by lick decision
                            arr[mouse + '_' + reg][i_tp, :] = average_fun(binary_truth=ps[np.intersect1d(np.where(lick == x)[0], inds_training)],
                                                                          estimate=pred_ps[np.intersect1d(np.where(lick == x)[0], inds_training)])
                        elif split_fourway is True:  # split two ways by lick decision
                            arr[mouse + '_' + reg][i_tp, :] = average_fun(binary_truth=ps[np.where(df_prediction_test[mouse]['outcome_test'] == x)[0]],
                                                                          estimate=pred_ps[np.where(df_prediction_test[mouse]['outcome_test'] == x)[0]])

                    for x, arr in ps_pred_split.items():
                        if split_fourway is False:  # split two ways by lick decision
                            arr[mouse + '_' + reg][i_tp, :] = [np.mean(pred_ps[lick == x]), np.std(pred_ps[lick == x])]
                        elif split_fourway is True:  # split two ways by lick decision
                            arr[mouse + '_' + reg][i_tp, :] = [np.mean(pred_ps[np.where(df_prediction_test[mouse]['outcome_test'] == x)[0]]), np.std(pred_ps[np.where(df_prediction_test[mouse]['outcome_test'] == x)[0]])]

                if 'angle_decoders' in df_prediction_train[mouse].columns:
                    angle_dec[mouse + '_' + reg][i_tp] = np.mean(df_prediction_train[mouse]['angle_decoders'])

    return (lick_acc, lick_acc_split, ps_acc, ps_acc_split, ps_pred_split, lick_half, angle_dec, decoder_weights)

## Create list with standard colors:
color_dict_stand = {}
for ii, x in enumerate(plt.rcParams['axes.prop_cycle']()):
    color_dict_stand[ii] = x['color']
    if ii > 8:
        break  # after 8 it repeats (for ever)



def wilcoxon_test(acc_dict):
    """Perform wilcoxon signed rank test for dictionoary of S1/S2 measurements. Each
    S1/S2 pair per mouse is a paired sample for the test. Perform test on each time point.

    Parameters:
    ----------------------
        acc_dict: dict
            dictionary of np.arrays of (n_timepoints x 2) or (n_timepoints)
            Wilcoxon test is performed across all mice, comparing regions, for each time point.

    Returns:
    ---------------------
        p_vals: np.array of size n_timepoints
            P values of W test.
    """
    reg_mouse_list = list(acc_dict.keys())
    mouse_list = np.unique([xx[:-3] for xx in reg_mouse_list])
    reg_list = ['s1', 's2']
    mouse_s1_list = [mouse + '_s1' for mouse in mouse_list]
    mouse_s2_list = [mouse + '_s2' for mouse in mouse_list]

    n_tp = acc_dict[reg_mouse_list[0]].shape[0]
    p_vals = np.zeros(n_tp)
    for tp in range(n_tp):
        if acc_dict[reg_mouse_list[0]].ndim == 2:
            s1_array = [acc_dict[ms1][tp, 0] for ms1 in mouse_s1_list]
            s2_array = [acc_dict[ms2][tp, 0] for ms2 in mouse_s2_list]
        elif acc_dict[reg_mouse_list[0]].ndim == 1:
            s1_array = [acc_dict[ms1][tp] for ms1 in mouse_s1_list]
            s2_array = [acc_dict[ms2][tp] for ms2 in mouse_s2_list]

        stat, pval = scipy.stats.wilcoxon(x=s1_array, y=s2_array, alternative='two-sided')
        p_vals[tp] = pval#.copy()
    return p_vals

def make_violin_df_custom(input_dict_df, flat_normalise_ntrials=False, verbose=0):
    """Function to turn my custom dictionary structure of DF to a single DF, suitable for a violin plot.

    Parameters:
    ---------------
        input_dict_df: dcit with structure [reg][tp][mouse]
            Assuming all mice/tp/reg combinations (regular grid)
        flat_normalise_ntrials: bool, default=False
            whether to normalise mice by number of trials. This can be required by the group averaging
            of decoders also ignores number of trials per mouse. If true, this is achieved by
            creating multiple copies of each mouse such that in the end each mouse has approximately
            the same number of trials
        verbose: int, default=0
            verbosity index

    Returns:
    ---------------
    new_df: dict with structure [tp]
    """
    dict_df = input_dict_df.copy()
    region_list = list(dict_df.keys())
    bool_two_regions = (region_list == ['s1', 's2'])
    if not bool_two_regions:
        assert (region_list == ['s1']) or (region_list == ['s2'])
    timepoints = list(dict_df[region_list[0]].keys())
    mouse_list = list(dict_df[region_list[0]][timepoints[0]].keys())
    n_multi = {}
    mouse_multi = 1
    ## add labels:
    for reg in region_list:
        for tp in timepoints:
            for mouse in mouse_list:
                dict_df[reg][tp][mouse]['region'] = reg.upper()
                dict_df[reg][tp][mouse]['mouse'] = mouse
                dict_df[reg][tp][mouse]['n_trials_mouse'] = len(dict_df[reg][tp][mouse])
                if flat_normalise_ntrials and mouse_multi:
                    n_multi[mouse] = np.round(10000 / len(dict_df[reg][tp][mouse])).astype('int')
                    assert n_multi[mouse] >= 10  # require at least 10 multiplications (so that relative error <10% for 1 mouse)
                    if verbose:
                        print(f'Number of trials for mouse {mouse}: {len(dict_df[reg][tp][mouse])}, multiplications: {np.round(10000 / len(dict_df[reg][tp][mouse]), 2)}')
            mouse_multi = 0 #  after first iteration
    ## Concatenate:
    new_df = {}
    for tp in timepoints:
        if flat_normalise_ntrials is False:
            if bool_two_regions:
                new_df[tp] = pd.concat([dict_df['s1'][tp][mouse] for mouse in mouse_list] +
                                       [dict_df['s2'][tp][mouse] for mouse in mouse_list])
            else:
                new_df[tp] = pd.concat([dict_df[region_list[0]][tp][mouse] for mouse in mouse_list])
        elif flat_normalise_ntrials:
            if bool_two_regions:
                new_df[tp] = pd.concat([pd.concat([dict_df['s1'][tp][mouse] for x in range(n_multi[mouse])]) for mouse in mouse_list] +
                                       [pd.concat([dict_df['s2'][tp][mouse] for x in range(n_multi[mouse])]) for mouse in mouse_list])
            else:
                new_df[tp] = pd.concat([pd.concat([dict_df[region_list[0]][tp][mouse] for x in range(n_multi[mouse])]) for mouse in mouse_list])
    if verbose:
        for mouse in mouse_list:
            print(f'Corrected number of trials for mouse {mouse}: {len(new_df[timepoints[0]][new_df[timepoints[0]]["mouse"] == mouse])}')
    return new_df

def difference_pre_post(ss, tt='hit', reg='s1', duration_window=1.2):
    """Compute difference df/f response between a post-stim window and a pre_stim
    baseline window average. Computes the average window acitivty per neuron and per
    trial, and then returns the average of the elementwise difference between
    all neurons and trials.
    #TODO: merge with dynamic equivalent
    Parameters:
    ---------------
        ss: Session
            session to evaluate
        xx: str, default='hit'
            trial type
        reg: str, default='s1'
            region
        duration_window_: float
            length of  window

    Returns:
    ---------------
        metric: float
            difference

    """
    inds_pre_stim = np.logical_and(ss.filter_ps_time < 0, ss.filter_ps_time >= -2) # hard-set 2 second pre stimulus baseline
    inds_post_stim = np.logical_and(ss.filter_ps_time < (ss.filter_ps_time[ss.filter_ps_time > 0][0] + duration_window),
                                    ss.filter_ps_time >= ss.filter_ps_time[ss.filter_ps_time > 0][0])  # post stimulus window

    if reg == 's1':
        reg_inds = ss.s1_bool
    elif reg == 's2':
        reg_inds = ss.s2_bool


    if tt == 'ur_hit':  # unrewarded hit
        general_tt = 'hit' # WILL be changed later, but this is more efficient I think with lines of code
        odd_tt_only = True
    elif tt == 'ar_miss':  # autorewarded miss
        general_tt = 'miss'
        odd_tt_only = True
    else:  # regular cases
        general_tt = tt
        if tt == 'hit' or tt == 'miss':
            odd_tt_only = False  # only use regular ones
        elif tt == 'fp' or tt == 'cr':
            odd_tt_only = None  # does not matter

    if odd_tt_only is None or general_tt == 'fp' or general_tt == 'cr': # if special tt do not apply
        pre_stim_act = ss.behaviour_trials[:, np.logical_and(ss.photostim < 2,
                                             ss.outcome==general_tt), :][:, :, ss.filter_ps_array[inds_pre_stim]][reg_inds, :, :]
        post_stim_act = ss.behaviour_trials[:, np.logical_and(ss.photostim < 2,
                                             ss.outcome==general_tt), :][:, :, ss.filter_ps_array[inds_post_stim]][reg_inds, :, :]
    elif general_tt =='miss' and odd_tt_only is not None:  # if specified miss type (autorewarded or not autorewarded)
        pre_stim_act = ss.behaviour_trials[:, np.logical_and.reduce((ss.photostim < 2,
                                                 ss.outcome==general_tt, ss.autorewarded==odd_tt_only)), :][:, :, ss.filter_ps_array[inds_pre_stim]][reg_inds, :, :]
        post_stim_act = ss.behaviour_trials[:, np.logical_and.reduce((ss.photostim < 2,
                                                 ss.outcome==general_tt, ss.autorewarded==odd_tt_only)), :][:, :, ss.filter_ps_array[inds_post_stim]][reg_inds, :, :]
    elif general_tt =='hit' and odd_tt_only is not None:  # if specified hit type (unrewarded or rewarded)
        if odd_tt_only is True:  # unrewarded hit
            general_tt = 'miss'   # unrewarded hits are registered as misssess
        pre_stim_act = ss.behaviour_trials[:, np.logical_and.reduce((ss.photostim < 2,
                                                 ss.outcome==general_tt, ss.unrewarded_hits==odd_tt_only)), :][:, :, ss.filter_ps_array[inds_pre_stim]][reg_inds, :, :]
        post_stim_act = ss.behaviour_trials[:, np.logical_and.reduce((ss.photostim < 2,
                                                 ss.outcome==general_tt, ss.unrewarded_hits==odd_tt_only)), :][:, :, ss.filter_ps_array[inds_post_stim]][reg_inds, :, :]

    pre_met = np.mean(pre_stim_act, 2)
    post_met = np.mean(post_stim_act, 2)

    metric = np.mean(post_met - pre_met)
    if metric.shape == ():  # if only 1 element it is not yet an array, while the separate trial output can be, so change for consistency
        metric = np.array([metric]) #  pack into 1D array
    return metric

def difference_pre_post_dynamic(ss, tt='hit', reg='s1', duration_window_pre=1.2,
                                tp_post=1.0, odd_tt_only=None, return_trials_separate=False):
    """Compute difference df/f response between a post-stim timepoint tp_post and a pre_stim
    baseline window average. Returns the average of the elementwise difference between
    all neurons and trials.

    Parameters:
    ---------------
        ss: Session
            session to evaluate
        tt: str, default='hit'
            trial type
        reg: str, default='s1'
            region
        duration_window_pre: float
            lenght of baseline window, taken <= 0
        tp_post: float
            post stim time point
        odd_tt_only: bool or None, default=None
            if True; only eval unrew_hit / autorew miss; if False; only evaluate non UH/AM; if None; evaluate all
            i.e. bundles boolean for unrewarded_hits and autorewarded trial types.

    Returns:
    ---------------
        metric: float
            difference

    """
    inds_pre_stim = np.logical_and(ss.filter_ps_time <= 0, ss.filter_ps_time >= (-1 * duration_window_pre))
    frame_post = ss.filter_ps_array[np.where(ss.filter_ps_time == tp_post)[0]]  # corresponding frame
    if reg == 's1':
        reg_inds = ss.s1_bool
    elif reg == 's2':
        reg_inds = ss.s2_bool

    if tt == 'ur_hit':  # unrewarded hit
        general_tt = 'hit' # WILL be changed later, but this is more efficient I think with lines of code
        odd_tt_only = True
    elif tt == 'ar_miss':  # autorewarded miss
        general_tt = 'miss'
        odd_tt_only = True
    elif tt =='spont_rew':
        odd_tt_only = True
        general_tt = tt
    else:  # regular cases
        general_tt = tt
        if tt == 'hit' or tt == 'miss':
            odd_tt_only = False  # only use regular ones
        elif tt == 'fp' or tt == 'cr':
            odd_tt_only = None  # does not matter
    if odd_tt_only is None or general_tt == 'fp' or general_tt == 'cr': # if special tt do not apply
        pre_stim_act = ss.behaviour_trials[:, np.logical_and(ss.photostim < 2,
                                             ss.outcome==general_tt), :][:, :, ss.filter_ps_array[inds_pre_stim]][reg_inds, :, :]
        post_stim_act = ss.behaviour_trials[:, np.logical_and(ss.photostim < 2,
                                             ss.outcome==general_tt), :][:, :, frame_post][reg_inds, :, :]
    elif general_tt =='miss' and odd_tt_only is not None:  # if specified miss type (autorewarded or not autorewarded)
        pre_stim_act = ss.behaviour_trials[:, np.logical_and.reduce((ss.photostim < 2,
                                                 ss.outcome==general_tt, ss.autorewarded==odd_tt_only)), :][:, :, ss.filter_ps_array[inds_pre_stim]][reg_inds, :, :]
        post_stim_act = ss.behaviour_trials[:, np.logical_and.reduce((ss.photostim < 2,
                                                 ss.outcome==general_tt, ss.autorewarded==odd_tt_only)), :][:, :, frame_post][reg_inds, :, :]
    elif general_tt =='hit' and odd_tt_only is not None:  # if specified hit type (unrewarded or rewarded)
        if odd_tt_only is True:  # unrewarded hit
            general_tt = 'miss'   # unrewarded hits are registered as misssess
        pre_stim_act = ss.behaviour_trials[:, np.logical_and.reduce((ss.photostim < 2,
                                                 ss.outcome==general_tt, ss.unrewarded_hits==odd_tt_only)), :][:, :, ss.filter_ps_array[inds_pre_stim]][reg_inds, :, :]
        post_stim_act = ss.behaviour_trials[:, np.logical_and.reduce((ss.photostim < 2,
                                                 ss.outcome==general_tt, ss.unrewarded_hits==odd_tt_only)), :][:, :, frame_post][reg_inds, :, :]

    elif tt == general_tt == 'spont_rew':
        flu_arr = ss.pre_rew_trials
        pre_stim_act = flu_arr[:, :, ss.filter_ps_array[inds_pre_stim]][reg_inds, :, :]
        post_stim_act = flu_arr[:, :, frame_post][reg_inds, :, :]

    else:
        raise ValueError('tt {} not understood'.format(tt))



    pre_met = np.squeeze(np.mean(pre_stim_act, 2))  # take mean over time points
    post_met = np.squeeze(post_stim_act)
    assert pre_met.shape == post_met.shape  # should now both be n_neurons x n_trials
    if return_trials_separate is False:
        metric = np.mean(post_met - pre_met)  # mean over neurons & trials,
    elif return_trials_separate:
        metric = np.mean(post_met - pre_met, 0)  # mean over neurons, not trials
    if metric.shape == ():  # if only 1 element it is not yet an array, while the separate trial output can be, so change for consistency
        metric = np.array([metric]) #  pack into 1D array
    return metric

def create_df_differences(sessions):
    ## Compute pre stim window vs post stim window
    dict_diff_wind = {name: np.zeros(8 * len(sessions), dtype='object') for name in ['diff_dff', 'region', 'trial_type', 'session']}
    ind_data = 0
    for _, sess in sessions.items():
        for reg in ['s1', 's2']:
            for tt in ['hit', 'fp', 'miss', 'cr']:
                mean_diff = difference_pre_post(ss=sess, 
                        tt=tt, reg=reg, duration_window=1)
                if len(mean_diff) == 0:
                    pass
                else:
                    dict_diff_wind['diff_dff'][ind_data] = mean_diff[0]
                    dict_diff_wind['region'][ind_data] = reg.upper()
                    dict_diff_wind['trial_type'][ind_data] = tt
                    dict_diff_wind['session'][ind_data] = sess.signature
                    ind_data += 1

    df_differences = pd.DataFrame(dict_diff_wind)        
    return df_differences

def create_df_dyn_differences(sessions, tp_dict):
    # list_tp = tp_dict['mutual'][np.where(np.logical_and(tp_dict['mutual'] >= -2, tp_dict['mutual'] <= 5))]
    list_tp = tp_dict['mutual'][np.where(tp_dict['mutual'] >= -2)[0]]
    list_tt = ['hit', 'fp', 'miss', 'cr', 'ur_hit', 'ar_miss']
    # dict_diff = {name: np.zeros(2 * len(list_tt) * len(sessions) * 
    #                             len(list_tp), dtype='object') for name in ['diff_dff', 'region', 'trial_type', 'session', 'timepoint', 'new_trial_id']}
    ind_data = 0
    dict_diff = {name: np.array([]) for name in ['diff_dff', 'region', 'trial_type', 'session', 'timepoint', 'new_trial_id']}  # initiate empy dicts
    for _, sess in tqdm(sessions.items()):
        for tp in list_tp:
            for reg in ['s1', 's2']:
                for tt in list_tt:
    #                 dict_diff['diff_dff'][ind_data] = pof.difference_pre_post_dynamic(ss=sess, 
    #                                           general_tt=tt, reg=reg, duration_window_pre=2, 
    #                                           tp_post=tp, odd_tt_only=True)
    #                 dict_diff['region'][ind_data] = reg.upper()
    #                 dict_diff['trial_type'][ind_data] = tt
    #                 dict_diff['session'][ind_data] = sess.signature
    #                 dict_diff['timepoint'][ind_data] = tp.copy()
    #                 ind_data += 1
                    mean_trials = difference_pre_post_dynamic(ss=sess, 
                                            tt=tt, reg=reg, duration_window_pre=2, 
                                            tp_post=tp, return_trials_separate=True)
                    if len(mean_trials) == 0:
    #                     print(tp, reg, sess, tt ,'   no trials')
                        pass
                    else:  # add array of new values
                        dict_diff['diff_dff'] = np.concatenate((dict_diff['diff_dff'], mean_trials))
                        dict_diff['region'] = np.concatenate((dict_diff['region'], [reg.upper() for x in range(len(mean_trials))]))
                        dict_diff['trial_type'] = np.concatenate((dict_diff['trial_type'], [tt for x in range(len(mean_trials))]))
                        dict_diff['session'] = np.concatenate((dict_diff['session'], [sess.signature for x in range(len(mean_trials))]))
                        dict_diff['timepoint'] = np.concatenate((dict_diff['timepoint'], [tp.copy() for x in range(len(mean_trials))]))
                        dict_diff['new_trial_id'] = np.concatenate((dict_diff['new_trial_id'], [ind_data + x for x in range(1, len(mean_trials) + 1)]))  # continuing indices
                        ind_data = dict_diff['new_trial_id'][-1]
    dict_diff['timepoint'] = dict_diff['timepoint'].astype('float32')
    dict_diff['diff_dff'] = dict_diff['diff_dff'].astype('float32')
    df_dyn_differences = pd.DataFrame(dict_diff)
    return df_dyn_differences

def get_decoder_data_for_violin_plots(sessions, tp_list=[1.0, 4.0]):
    ## Which time points to include in violin plots:
      # in seconds

    region_list = ['s1', 's2']
    dict_df_test = {reg: {} for reg in region_list}
    for reg in region_list:
        for tp in tp_list:  # retrain (deterministic) decoders for these time points, and save detailed info
            _, dict_df_test[reg][tp], __, ___ = train_test_all_sessions(sessions=sessions, verbose=0,# n_split=n_split,
                                        trial_times_use=np.array([tp]), return_decoder_weights=False,
                                        hitmiss_only=False,# list_test=['dec', 'stim'],
                                        include_autoreward=False, neurons_selection=reg,
                                        C_value=50, reg_type='l2', train_projected=False)

    ## turn into df that can be used for violin plots efficiently,
    ## normalised so that each animals is equally important in averaging
    violin_df_test = make_violin_df_custom(input_dict_df=dict_df_test, 
                                            flat_normalise_ntrials=True, verbose=1) 
    return violin_df_test


def create_df_table_details(sessions):
    """Create Dataframe table with details of sessions."""
    n_sessions = len(sessions)
    column_names = ['Mouse', 'Run', 'f (Hz)', #'# Imaging planes',
                    r"$N$" + 'S1', r"$N$" + 'S2',
                    'Trials', 'Hit', 'FP', 'Miss', 'CR', 'UR Hit', 'AR Miss', 'Too early']
    dict_details = {cc: np.zeros(n_sessions, dtype='object') for cc in column_names}
    for key, ss in sessions.items():
        dict_details['Mouse'][key] = ss.mouse
        dict_details['Run'][key] = ss.run_number
        dict_details['f (Hz)'][key] = ss.frequency
#         dict_details['# Imaging planes'][key] = len(np.unique(ss.plane_number))
        dict_details[r"$N$" + 'S1'][key] = np.sum(ss.s1_bool)
        dict_details[r"$N$" + 'S2'][key] = np.sum(ss.s2_bool)
        leave_out_150_inds = ss.photostim < 2
        dict_details['Trials'][key] = len(ss.outcome[leave_out_150_inds])
        print(ss.name, ss.n_trials, len(ss.outcome))
        dict_details['Hit'][key] = np.sum(ss.outcome[leave_out_150_inds] == 'hit')
        dict_details['FP'][key] = np.sum(ss.outcome[leave_out_150_inds] == 'fp')
        dict_details['Miss'][key] = np.sum(np.logical_and(ss.outcome[leave_out_150_inds] == 'miss',
                                                          ss.autorewarded[leave_out_150_inds] == False))
        dict_details['CR'][key] = np.sum(ss.outcome[leave_out_150_inds] == 'cr')
        dict_details['Too early'][key] = np.sum(ss.outcome[leave_out_150_inds] == 'too_')
        dict_details['UR Hit'][key] = np.sum(ss.unrewarded_hits[leave_out_150_inds])
        dict_details['AR Miss'][key] = np.sum(ss.autorewarded[leave_out_150_inds])
        assert np.sum([dict_details[xx][key] for xx in ['Hit', 'FP', 'Miss', 'CR', 'Too early', 'AR Miss']]) == dict_details['Trials'][key], 'total number of trials is not correct'
    df_details = pd.DataFrame(dict_details)
    df_details = df_details.sort_values(by=['Mouse', 'Run'])
    df_details = df_details.reset_index()
    del df_details['index']
    return df_details

def perform_logreg_cv(sessions, c_value_array, reg_list=['s1', 's2']):
    """Takes max over regions if both are given"""
    max_acc_scores = {}
    for key, ss in sessions.items():
        print(ss)
        ## First without reg:
        max_dec_values = np.zeros((len(c_value_array) + 1, 2))
        (lick_acc, lick_acc_split, ps_acc, ps_acc_split, ps_pred_split, lick_half, 
                 angle_dec, dec_weights) = compute_accuracy_time_array(sessions={0: ss}, time_array=tp_dict['cv_reg'],
                                                              projected_data=False, reg_type='none',
                                                              region_list=reg_list,
                                                              average_fun=class_av_mean_accuracy)
        assert len(lick_acc) == 1
        mr_name = list(lick_acc.keys())[0]
        max_lick_dec = np.max(lick_acc[mr_name])  #np.max([np.max(reg_acc[:, 0]) for _, reg_acc in lick_acc.items()])
        max_ps_dec = np.max(ps_acc[mr_name])  #np.max([np.max(reg_acc[:, 0]) for _, reg_acc in ps_acc.items()])
        max_dec_values[0, :] = max_lick_dec.copy(), max_ps_dec.copy()  
        ## Then with varying reg strengths:
        for i_c, c_value in enumerate(c_value_array):
            ## Compute results
            (lick_acc, lick_acc_split, ps_acc, ps_acc_split, ps_pred_split, lick_half, 
                 angle_dec, dec_weights) = compute_accuracy_time_array(sessions={0: ss}, time_array=tp_dict['cv_reg'],
                                                              projected_data=False, 
                                                              reg_type='l2', regularizer=c_value,
                                                              region_list=reg_list,
                                                              average_fun=class_av_mean_accuracy)
            assert len(lick_acc) == 1
#             mr_name = list(lick_acc.keys())[0]
            max_lick_dec = np.max(lick_acc[reg][:, 0]) # np.max(lick_acc[mr_name])  #np.max([np.max(reg_acc[:, 0]) for _, reg_acc in lick_acc.items()])
            max_ps_dec = np.max(ps_acc[reg][:, 0]) #np.max(ps_acc[mr_name])  #np.max([np.max(reg_acc[:, 0]) for _, reg_acc in ps_acc.items()])
            max_dec_values[i_c + 1, :] = max_lick_dec.copy(), max_ps_dec.copy()
        ## Save:
        max_acc_scores[key] = max_dec_values.copy()
    return max_acc_scores

def create_tp_dict(sessions):
    ## Integrate different imaging frequencies to create an array of mutual (shared) time points:
    freqs = np.unique([ss.frequency for _, ss in sessions.items()])
    tp_dict = {}
    for ff in freqs:
        for _, ss in sessions.items():   # assume pre_seconds & post_seconds equal for all sessions
            if ss.frequency == ff:
                tp_dict[ff] = ss.filter_ps_time
    if len(freqs) == 2:  # for hard-coded bit next up
        tp_dict['mutual'] = np.intersect1d(ar1=tp_dict[freqs[0]], ar2=tp_dict[freqs[1]])
    elif len(freqs) == 1:
        tp_dict['mutual'] = tp_dict[freqs[0]]
    return tp_dict

def opt_leaf(w_mat, dim=0, link_metric='correlation'):
    '''create optimal leaf order over dim, of matrix w_mat.
    see also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.optimal_leaf_ordering.html#scipy.cluster.hierarchy.optimal_leaf_ordering'''
    assert w_mat.ndim == 2
    if dim == 1:  # transpose to get right dim in shape
        w_mat = w_mat.T
    dist = scipy.spatial.distance.pdist(w_mat, metric=link_metric)  # distanc ematrix
    link_mat = scipy.cluster.hierarchy.ward(dist)  # linkage matrix
    if link_metric == 'euclidean' and True:
        opt_leaves = scipy.cluster.hierarchy.leaves_list(scipy.cluster.hierarchy.optimal_leaf_ordering(link_mat, dist))
        print('OPTIMAL LEAF SOSRTING AND EUCLIDEAN USED')
    else:
        opt_leaves = scipy.cluster.hierarchy.leaves_list(link_mat)
    return opt_leaves, (link_mat, dist)