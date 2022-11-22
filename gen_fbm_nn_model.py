#Python script to train, evaluate and save model to estimate the Hurst exponent from trajectory of fBm.
# import tensorflow as tf
import numpy as np
import tensorflow as tf
from stochastic.processes.continuous import FractionalBrownianMotion

def gen_fbm_data(nsamples,ntimes):
    """ 
    Function to produce fractional brownian motion data for neural network 
    training and testing
    Inputs: nsamples; number of samples, ntimes; number of times
    Outputs: traindata; training data for NN, trainlabels; labels associated 
            with traindata
    """
    data = np.empty((nsamples,ntimes))
    labels = np.empty((nsamples,1))
    for i in range(0,nsamples):
        hurst_exp = np.random.uniform(0.,1.)
        fbm = FractionalBrownianMotion(hurst=hurst_exp,t=1,rng=None)
        x = fbm.sample(ntimes)
        #apply differencing and normalization on the data
        dx = (x[1:]-x[0:-1])/(np.amax(x)-np.amin(x))
        data[i,:] = dx
        labels[i,:] = hurst_exp
        
    return data,labels
 
def gen_nn_model(ntimes,activation_func='relu',optimizer='adam',
                 loss_func='mean_absolute_error',summary=1):
    
    #create the model for a fully-connected network
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(ntimes,activation=activation_func,input_shape=(ntimes,)),
        tf.keras.layers.Dense(ntimes-1,activation=activation_func),
        tf.keras.layers.Dense(ntimes-2,activation=activation_func),
        tf.keras.layers.Dense(1,activation=activation_func)])
    #add optimizer, a loss function and metrics#
    # optimizer = tf.keras.optimizers.RMSprop(0.001)
    nn_model.compile(optimizer=optimizer,
                  loss=loss_func,
                  metrics=['mean_absolute_error','mean_squared_error'])
    if summary==1:
        nn_model.summary()
    return nn_model

def train_nn_model(nn_model,training_data,training_labels,ntimes,
                   n_epochs,validation_split=0.6,verbose=1):
    
    history = nn_model.fit(training_data,training_labels,epochs=n_epochs,
                        validation_split=validation_split,verbose=verbose)
    
    print("Saving model")
    nn_model.save("./model3dense_n"+str(ntimes)+".h5")
    del nn_model

def evaluate_model(nsamples,ntimes,verbose=1):

    #evaluate the model generalizes by using the test data set
    testdata, testlabels = gen_fbm_data(nsamples, ntimes)
    model = tf.keras.models.load_model("./model3dense_n"+str(ntimes)+".h5")
    loss, mean_abs_error, mean_squared_error = model.evaluate(testdata, testlabels, verbose=verbose)
    with open('metrics_'+"model3dense_n"+str(ntimes), 'w') as f:
        print("{}   #Testing set Loss".format(loss), file=f)
        print("{}   #Testing set Mean Abs Error".format(mean_abs_error), file=f)
        print("{}   #Testing set Mean Squared Error".format(mean_squared_error), file=f)

def __main__(nsamples,ntimes,epochs):

    training_data,training_labels = gen_fbm_data(nsamples,ntimes)
    model = gen_nn_model(ntimes)
    train_nn_model(model,training_data,training_labels,ntimes,epochs)
    evaluate_model(nsamples,ntimes)