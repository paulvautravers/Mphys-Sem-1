# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 17:26:13 2022

@author: PaulV
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from scipy.stats import norm
from sklearn.mixture import GaussianMixture as GMM
from scipy import stats
import json
import gen_fbm_nn_model as fbm_nn

def load_nn_model(window_size,n_samples=10000,n_epochs=100):

    try:
        model = tf.keras.models.load_model("model3dense_n{}.h5".format(window_size))
    except OSError:
        fbm_nn.__main__(n_samples,window_size,n_epochs)
        model = tf.keras.models.load_model("model3dense_n{}.h5".format(window_size))
    return model 

def find_displacement(x_data, y_data, z_data, start_index=0):

    disps = np.sqrt(np.power(x_data-x_data[0],2) + np.power(y_data-y_data[0],2) + np.power(z_data-z_data[0],2))
    
    return disps 

def estimate_hurst(disps, time, window):
    
    h = np.array([])
    ht = np.array([])
    for i in range(int(window/2), len(disps)-(1+int(window/2))):
        if window % 2 == 1:  # odd window size
            inx = disps[(i-int(window/2)):(i+2+int(window/2))]
        else:  # even window size
            inx = disps[(i-int(window/2)):(i+1+int(window/2))]
        #apply differencing and normalization on the data
        inx = np.array([(inx[1:]-inx[0:-1])/(np.amax(inx)-np.amin(inx))])
        test = model.predict(inx,verbose=0)
        h=np.append(h,test[0][0])
        ht = np.append(ht,time[i])
        
    return h,ht

def downsample(data_input, down_int, start_index=0):
    
    data_out = data_input.iloc[start_index::down_int]
    return data_out


def dsample_est_hurst(data_in, ds_rate, window):
    """
    Args:
        data_in: pandas dataframe
        ds_rate: int, downsampling step size
    Returns:
        h_arr: 2D np array
        ht_arr: 2D np array
    """
    
    if window % 2 == 1:  # for odd window size
        h_arr = np.empty((ds_rate,(len(data_in)//ds_rate)-(window)))
        ht_arr = np.empty((ds_rate,(len(data_in)//ds_rate)-(window)))
    else:  # for even window size
        h_arr = np.empty((ds_rate,(len(data_in)//ds_rate)-(window+1)))
        ht_arr = np.empty((ds_rate,(len(data_in)//ds_rate)-(window+1)))
    
    for i in np.arange(ds_rate):
        
        downsampled_data = downsample(data_in, ds_rate, i)
        x = np.array(downsampled_data['Position X'])
        y = np.array(downsampled_data['Position Y'])
        z = np.array(downsampled_data['Position Z'])
        t = np.array(downsampled_data['Absolute Time'])
        displacements = find_displacement(x,y,z)
        h,ht = estimate_hurst(displacements, t, window)
        
        if len(h)>np.shape(h_arr)[1]:
            h=h[:-1]
            ht=ht[:-1]
            
        h_arr[i] = h
        ht_arr[i] = ht
        
    return h_arr,ht_arr

def filter_data(data_in, max_step_size, window, restriction=10000):
    """
    Only keeps data for tracks that are long enough for hurst exponent estimation at a given downsampling step size.
    Args: 
        data_in: pandas dataframe, original data
        max_step_size: int, maximum downsampling step size
        window: int, size of rolling window for hurst component estimation
    Returns:
        filtered_data: pandas dataframe
    """
    if window % 2 == 1:  # for odd window size
        tracks_to_keep = data_in.TrackID.value_counts().loc[lambda x: (x//max_step_size) > (window)].reset_index()['index']
    else:  # for even window size
        tracks_to_keep = data_in.TrackID.value_counts().loc[lambda x: (x//max_step_size) > (window+1)].reset_index()['index']
    
    if restriction<len(tracks_to_keep):
        filtered_data = data_in[data_in['TrackID'].isin(tracks_to_keep[:restriction])]
    else:
        filtered_data = data_in[data_in['TrackID'].isin(tracks_to_keep)]
    return filtered_data

def get_h_values(filtered_data, step_size, window, restriction):
    """
    Args:
        filtered_data: pandas dataframe
    Returns 1D array of (mean) hurst exponent values for a given step size.
    """
    track_id_values = np.unique(filtered_data['TrackID'])
    
    h = np.array([])
    for tid in track_id_values:
        print('Track: {}'.format(tid))
        track_data = filtered_data[filtered_data['TrackID']==tid]
        h_arr, ht_arr = dsample_est_hurst(track_data, step_size, window)  
        # h_av_arr, ht_av_arr = average_hurst(h_arr, ht_arr)  
        h = np.append(h, np.ravel(h_arr))
        
    return h

def get_h_dict(filtered_data, step_sizes, window, restriction):
    """
    """
    h_dict = {}
    
    for i, s in enumerate(step_sizes):
        print('step size: {}'.format(s))
        h_arr = get_h_values(filtered_data, s, window, restriction)
        h_dict["{}".format(s)] = h_arr.tolist()
        
    return h_dict

def get_hist_h(h_dictionary, nbins):
    """
    """
    keys_list = list(h_dictionary)
    
    counts_all = np.empty((len(keys_list), nbins))
    bins_all = np.empty((len(keys_list), nbins+1))
    
    for i,key in enumerate(keys_list):
        counts, bins = np.histogram(h_dictionary[key], nbins, density=True)  # normalised so area under histogram is 1
        counts_all[i] = counts
        bins_all[i] = bins
    
    return counts_all, bins_all

def save_h_data(h_data, file_name, window, step_sizes, restriction):
    """
    """
    with open('h_dict_'+'w{}_'.format(window)+'s{}_'.format(max(step_sizes))+'r{}_'.format(restriction)+file_name, 'w') as file:
        file.write(json.dumps(h_data))
        
def get_dict_from_file(file_name, window, step_sizes, restriction):
    """
    Reads in dictionary of H values.
    """
    with open('h_dict_'+'w{}_'.format(window)+'s{}_'.format(max(step_sizes))+'r{}_'.format(restriction)+file_name, 'r') as file:
        h_dict = file.read()
    
    return json.loads(h_dict)

def number_of_tracks(filenames, window, step_sizes=np.array([1,2,3,4,5,6,7,8,9,10]), pandas_df=True):
    """
    """
    ntracks_dict = {}
    
    for i, file in enumerate(filenames):
        #data = pd.read_csv('haemocyte_tracking_data/'+file+'.csv')
        data = pd.read_csv(file+'.csv')
        ntracks = np.empty((len(step_sizes)+1))
        ntracks[0] = len(np.unique(data['TrackID']))  # number of tracks in original data (independent of window, step size)
        for j, s in enumerate(step_sizes):
            filtered_data = filter_data(data, s, window)
            ntracks[j+1] = len(np.unique(filtered_data['TrackID']))  # number of tracks in filtered data
        ntracks_dict[file] = ntracks.tolist()
    
    if not pandas_df:
        return ntracks_dict
    
    else:
        column_labels = np.array(['original'])
        column_labels = np.append(column_labels, step_sizes.astype(str))
        df = pd.DataFrame.from_dict(ntracks_dict, orient='index', columns=column_labels)
        return df.apply(pd.to_numeric, downcast='integer')
    
def gen_h_dict_all_files(filenames,step_sizes,window,restriction):
    for i, file in enumerate(filenames):
        print('opened file {}'.format(file))
        #data = pd.read_csv('haemocyte_tracking_data/' + file + '.csv')
        data = pd.read_csv(file + '.csv')
        filtered_data = filter_data(data, max(step_sizes), window, restriction)
        h_dict = get_h_dict(filtered_data, step_sizes, window, restriction)
        save_h_data(h_dict, file, window, step_sizes, restriction)
    
### Main #################

window = 13
ds_steps = np.array([1,2,3,4,5,6,7,8])
restriction = 30

model = load_nn_model(window)

filenames = np.array(['Control_frame001-200','Control_frame200-400',
                      'Control_frame400-600','Control_frame600-800',
                      'Control_frame800-1000','Control_frame1000-1200',
                      'LanB1_frame001-200','LanB1_frame200-400',
                      'LanB1_frame400-600','LanB1_frame600-800',
                      'LanB1_frame800-1000','LanB1_frame1000-1200',
                      'defLanB1_300817_frame200-400',
                      'defLanB1_300817_frame400-600'])

print('for a window size of {}:'.format(window))
print(filenames[:2])
number_of_tracks(filenames, window)

gen_h_dict_all_files(filenames[:2],ds_steps,window,restriction)