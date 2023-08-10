import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib as mpl
# Disable tensorflow info, warning and error messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

#########################################################################################
###################################  Preprocess data  ###################################
#########################################################################################

def normalise_set(train, val):
    # Transform into uniform distribution between 0 and 1 using quantiles statistics
    scaler_quant  = QuantileTransformer().fit(train)
    train         = scaler_quant.transform(train)
    val           = scaler_quant.transform(val)
    # Next, transform the distribution to zero mean and unit variance
    scaler_stand  = StandardScaler().fit(train)
    train         = scaler_stand.transform(train)
    val           = scaler_stand.transform(val)

    # Return the transformed set as well as the transformations for inversing later
    return train, val, scaler_quant, scaler_stand


# Inverse normalise the training and validation data based on previous normalisation
def inverse_normalise_set(train, val, scaler_quant, scaler_stand):
    # First, inverse standard scaler
    train = scaler_stand.inverse_transform(train)
    val   = scaler_stand.inverse_transform(val)
    # Next, inverse quantiles transformation
    train = scaler_quant.inverse_transform(train)
    val   = scaler_quant.inverse_transform(val)

    # Return the set in its original scale
    return train, val


# Normalise a single array with quantiles and standard transformers like normalise_set()
def normalise_array(data, scaler_quant, scaler_stand):
    data = scaler_quant.transform(data)
    data = scaler_stand.transform(data)

    # Returned the transformed array
    return data

# Normalise a single array with quantiles and standard transformers like normalise_set()
def normalise_array_no_previous(data):
    scaler_quant  = QuantileTransformer().fit(data)
    data = scaler_quant.transform(data)
    scaler_stand  = StandardScaler().fit(data)
    data = scaler_stand.transform(data)

    # Returned the transformed array
    return data, scaler_quant, scaler_stand



# Inverse normalisation of a single array based on previous normalisation
def inverse_normalise_array(data, scaler_quant, scaler_stand):
    data = scaler_stand.inverse_transform(data)
    data = scaler_quant.inverse_transform(data)
    
    # Return the array in original scale
    return data





def main():

    mpl.style.use('seaborn-v0_8-whitegrid')

    #########################################################################################
    ################################# Load and normalise data ##############################
    #########################################################################################

    # Load data sets to base the normalisation transformations on
    X_norm = np.loadtxt('../data_dirs/3_params_1e6_points/parameters.dat')
    Y_norm = np.loadtxt('../data_dirs/3_params_1e6_points/rates.dat')
    

    # Get transformations based on the data set that the ANN has trained on
    _, scaler_quant_X, scaler_stand_X = normalise_array_no_previous(X_norm)
    _, scaler_quant_Y, scaler_stand_Y = normalise_array_no_previous(Y_norm)
    

    #########################################################################################
    ###################################  Train the network  #################################
    #########################################################################################

    model = tf.keras.models.load_model('model_eft_1e6')
    model.summary()


    #########################################################################################
    ###################################  Analyse results  ###################################
    #########################################################################################

    # Load verification data sets
    X = np.loadtxt('../data_dirs/3_params_1e3_points/parameters.dat')
    Y = np.loadtxt('../data_dirs/3_params_1e3_points/rates.dat')

    # Normalise the inputs and predict rates
    X = normalise_array(X, scaler_quant_X, scaler_stand_X)
    rates = model.predict(X)
    # Inverse normalise the rates
    rates = inverse_normalise_array(rates, scaler_quant_Y, scaler_stand_Y)
    # Save in file as in use case, just for speed measurements
    np.savetxt("test_rates.npy", rates)

    
    # __WARNING__ DO NOT TIME THE FOLLOWING
    
    #Q = 4
    
    error_means = []
    error_stds = []
    error_medians = []
    """
    outliers = np.array([0, 0, 0, 0, 0, 1, 1, 0, 6,1, 1])
    """
    #n_Q = 1
    n_Q = 10
    for Q in range(n_Q):
        #Q = 2
        rates_Q = rates[:,Q]
        Y_Q = Y[:,Q]

        n_outliers = 0
        error = rates_Q - Y_Q
        perc_error = []
        for i in range(len(Y_Q)):
            if Y_Q[i] == 0:
                print("Removed data with error " + str(error[i]))
            else:
                if error[i]/Y_Q[i] >= 10:
                    n_outliers += 1
                if Y_Q[i] != 0 and error[i]/Y_Q[i] < 10:
                    perc_error.append(error[i]/Y_Q[i])
        
        error_means.append(np.mean(perc_error))
        error_stds.append(np.std(perc_error))
        error_medians.append(np.median(perc_error))
        #print("Mean:"+str(mean)+" Std: "+str(std)+" Median: "+str(median)+" Outliers: "+str(n_outliers))
    error_means = np.array(error_means)
    error_stds = np.array(error_stds)
    error_medians = np.array(error_medians)
    
        
    '''
    # Using model_eft_1e6
    plt.axvline(error_medians, color="k", linestyle=":", label=r"Median")
    plt.axvline(error_means, color="k", label=r"$\mu$")
    plt.axvline(error_means-error_stds, color="k", linestyle="--", label=r"$\mu\pm\sigma$")
    plt.axvline(error_means+error_stds, color="k", linestyle="--")
    plt.xlabel("Error for predicted rates")
    plt.ylabel("Frequency")
    plt.legend()
    plt.hist(perc_error, bins=100, density=True)
    plt.savefig("../../../report_figures/eft_error_distribution.png", dpi=1200)
    plt.show()
    '''
        
    
    # Plot the error as a function of Q bins: For 10^3 data points, NN trained on 10^6
    # Using model_eft_1e6
    Qs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # Following data made for 0s removed and no data over 5 times larger, when model trained on 1e6
    #error_means = np.array([-0.091, -0.14, -0.11, -0.019, 0.084, 0.064, -0.028, -0.107, -0.020, 0.015])
    #error_stds = np.array([0.31, 0.21, 0.22, 0.18, 0.22, 0.21, 0.20, 0.24, 0.37, 0.36])
    #error_medians = np.array([-0.11, -0.11, -0.010, -0.013, -0.048, 0.032, -0.017, -0.14, -0.04,-0.010])
    print("BLAAAAAAAAAAAAAAAAA")
    print(error_means)
    print(error_stds)
    print(error_medians)

    plt.plot(Qs, error_medians, label="Median")
    plt.plot(Qs, error_means, color="k", linestyle="-", label=r"$\mu$")
    plt.plot(Qs, error_means-error_stds, color="k", linestyle="--", label=r"$\mu\pm\sigma$")
    plt.plot(Qs, error_means+error_stds, color="k", linestyle="--")
    plt.legend()
    plt.xlabel(r"$Q$")
    plt.ylabel(r"Error for predicted rates")
    plt.savefig("../../../report_figures/eft_error_mean_std.png", dpi=1200)
    plt.show()
    #'''



if __name__ == "__main__":
    main()
