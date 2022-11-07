# Mphys-Sem-1
Mila Jelic and Paul Vautravers
Repository for Mphys project investigation into anomalous transport of haemocytes in drosophila embryos.

Contents:

Folders:

- archive: Folder with all old code, some may still be of use, but mostly code that has been replaced or no longer of interest
- h_dict_data: Dictionaries containing Hurst exponent series for different downsampling rates
- haemocyte_tracking_data: Input data to retrieve H values, from tom Millard

Code:

- Random Walk.ipynb: Initial code simulating random walks, should be moved to archive 
- fbm_random_walk.ipynb: Random walk simulations using fractional brownian motion and multifractional brownian motion. 
- gen_fbm_nn_model.py: Python script to generate trained neural network off of simulated fractional brownian motion
- get_h_data_haemocyte_fbm.py: python script to collect hurst dictionary data, imports python script above for training model if there isn't one already trained
- haemocyte_fbm.ipynb: jupyter version of get_h_data..py, contains a lot of scrap code 
- haemocyte_plot.ipynb: Contains code to plot 2d and 3d histograms of the hurst exponent
- mixture_modelling.ipynb: Code to perform gaussian/skew mixture modelling of hurst histograms, should focus on bounded distributions

