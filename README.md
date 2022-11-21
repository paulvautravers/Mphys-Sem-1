# Mphys-Sem-1
Mila Jelic and Paul Vautravers
Repository for Mphys project investigation into anomalous transport of haemocytes in drosophila embryos.

Contents:

Folders:

- archive: Folder with all old code, some may still be of use, but mostly code that has been replaced or no longer of interest
- h_dict_data: Dictionaries containing Hurst exponent series for different downsampling rates
- haemocyte_restructured: Contains 3 new scripts to collect H analysis data; get displacement data, get single step dictionaries and combine all dictionaries together
- haemocyte_tracking_data: Input data to retrieve H values, from tom Millard

Code:

- Random Walk.ipynb: Initial code simulating random walks, should be moved to archive 
- fbm_random_walk.ipynb: Random walk simulations using fractional brownian motion and multifractional brownian motion.
- gen_fbm_nn_model.py: Python script to generate trained neural network off of simulated fractional brownian motion
- get_correlation_data.ipynb: Script to calculate cos(theta) value across tracks and datasets, providing indication of how corellated the movement is
- get_h_data_haemocyte_fbm.py: python script to collect hurst dictionary data, imports python script above for training model if there isn't one already trained
-get_msd_and_persistence.ipynb: combined script for mean squared displacement as well as persistence parameter analysis 
- histogram_analysis.ipynb: script with plotting functions, as well as mean, variance etc and mixture modelling
- probability_fitting.ipynb: Script with skew, log and standard normal fitting of histogram data
- mixture_modelling.ipynb: Code to perform gaussian/skew mixture modelling of hurst histograms, should focus on bounded distributions

21/11/2022
