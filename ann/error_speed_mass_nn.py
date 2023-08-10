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

    # Load data for transformations
    X_norm = np.loadtxt('../data_dirs/1_param_1e5_points/parameters.dat')
    Y_norm = np.loadtxt('../data_dirs/1_param_1e5_points/rates.dat')
    X_norm = X_norm.reshape(-1,1)
    

    # Get transformations
    _, scaler_quant_X, scaler_stand_X = normalise_array_no_previous(X_norm)
    _, scaler_quant_Y, scaler_stand_Y = normalise_array_no_previous(Y_norm)

    #########################################################################################
    #####################################  Get the network  #################################
    #########################################################################################

    model = tf.keras.models.load_model('model_mass_1e5')
    model.summary()


    #########################################################################################
    ###################################  Analyse results  ###################################
    #########################################################################################

    # Load data to verify with
    X = np.loadtxt('../data_dirs/1_param_1e3_points/parameters.dat')
    Y = np.loadtxt('../data_dirs/1_param_1e3_points/rates.dat')
    X = X.reshape(-1,1)
    X = normalise_array(X, scaler_quant_X, scaler_stand_X)
    # X is normalised
    rates = model.predict(X)
    # Inverse normalise the rates
    rates = inverse_normalise_array(rates, scaler_quant_Y, scaler_stand_Y)
    # Save in file as in use case
    np.savetxt("test_rates.npy", rates)


    # __WARNING__ DO NOT TIME THE FOLLOWING
    
    
    error_means = [] 
    error_stds = [] 
    error_medians = [] 
    for Q in range(10):
        rates_Q = rates[:,Q]
        Y_Q = Y[:,Q]

        n_outliers = 0
        error = rates_Q - Y_Q
        perc_error = []

        for i in range(len(Y_Q)):
            err = error[i]/Y_Q[i]
            if Y_Q[i] == 0:
                print("Removed data with error " + str(error[i]))
            if err > 10:
                n_outliers += 1
            if Y_Q[i] != 0 and err < 10:
                perc_error.append(err)

        error_means.append(np.mean(perc_error))
        error_medians.append(np.median(perc_error))
        error_stds.append(np.std(perc_error))
        #print("Mean:"+str(mean)+" Std: "+str(std)+" Median: "+str(median)+" N outliers: "+str(n_outliers))
    
    """
    # Using model_eft_1e5
    plt.hist(perc_error, bins=100, density=True)
    plt.axvline(mean, color="k", label=r"$\mu$")
    plt.axvline(mean-std, color="k", linestyle="--", label=r"$\mu\pm\sigma$")
    plt.axvline(mean+std, color="k", linestyle="--")
    plt.axvline(median, color="k", linestyle=":", label=r"Median")
    plt.xlabel("Error for predicted rates")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig("../../../report_figures/error_distribution.png", dpi=1200)
    plt.show()
    """

    # Plot the error as a function of bins: For 10^3 data points, NN trained on 10^6
    # Using model_mass_1e5
    Qs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # Following data made for 0s removed and no data over 10 times larger
    error_means = np.array([-0.028, -0.01, -0.013, -0.098, 0.027, -0.068, -0.023, -0.010, -0.18, -0.059])
    error_stds = np.array([0.13, 0.57, 0.30, 0.27, 0.66, 0.95, 0.84, 1.05, 0.69, 0.93])
    error_medians = np.array([-0.006, -0.13, -0.048, -0.15, -0.15, -0.25, -0.22, -0.33, -0.26, -0.22])
    error_means = np.array(error_means)
    error_stds = np.array(error_stds)
    error_medians = np.array(error_medians)
    plt.plot(Qs, error_medians, label="Median")
    plt.plot(Qs, error_means, color="k", linestyle="-", label=r"$\mu$")
    plt.plot(Qs, error_means-error_stds, color="k", linestyle="--", label=r"$\mu\pm\sigma$")
    plt.plot(Qs, error_means+error_stds, color="k", linestyle="--")
    plt.legend()
    plt.xlabel(r"$Q$")
    plt.ylabel(r"Error for predicted rates")
    plt.savefig("../../../report_figures/error_mean_std.png", dpi=1200)
    plt.show() 

    # Plot the error as a function of the number of data points that the ANN has trained on,
    # for different Qs. Use median as its more robust to outliers
    '''
    # Following data made for 0s removed and no data over 20 times larger
    data_sizes = np.array([10**4, 10**5, 10**6])
    plt.plot(data_sizes, median_1, label=r"$Q=1$")
    plt.plot(data_sizes, median_2, label=r"$Q=2$")
    plt.plot(data_sizes, median_3, label=r"$Q=3$")
    plt.plot(data_sizes, median_4, label=r"$Q=4$")
    plt.plot(data_sizes, median_5, label=r"$Q=5$")
    plt.plot(data_sizes, median_6, label=r"$Q=6$")
    plt.plot(data_sizes, median_7, label=r"$Q=7$")
    plt.plot(data_sizes, median_8, label=r"$Q=8$")
    plt.plot(data_sizes, median_9, label=r"$Q=9$")
    plt.plot(data_sizes, median_10, label=r"$Q=10$")
    plt.legend()
    plt.xlabel(r"Data size used by the ANN")
    plt.ylabel(r"Error for predicted rates")
    #plt.savefig("../../../report_figures/error_mean_std.png", dpi=1200)
    '''






if __name__ == "__main__":
    main()
