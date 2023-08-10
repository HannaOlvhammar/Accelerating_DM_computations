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


    
    # Load data to verify with
    X_verify = np.loadtxt('../data_dirs/1_param_1e3_points/parameters.dat')
    Y_verify = np.loadtxt('../data_dirs/1_param_1e3_points/rates.dat')

    X_verify = X_verify.reshape(-1,1)
    # Transform the loguniform distribution of data into uniform distribution with
    # zero mean and unit variance
    _, X_verify, scaler_quant_X, scaler_stand_X = normalise_set(X_verify, X_verify)
    _, Y_verify, scaler_quant_Y, scaler_stand_Y = normalise_set(Y_verify, Y_verify)


    # Record properties of the data
#    val_data_size = np.shape(X_val)[0]
#    n_Q_values = np.shape(Y_val[1])[0]


    #########################################################################################
    ###################################  Train the network  #################################
    #########################################################################################

    # Trained on 1e5 points
    model = tf.keras.models.load_model('model_mass')
    model.summary()


    #########################################################################################
    ###################################  Analyse results  ###################################
    #########################################################################################


    rates = model.predict(X_verify)
    rates = inverse_normalise_array(rates, scaler_quant_Y, scaler_stand_Y)
    np.savetxt("test_rates.npy", rates)

    """
    # REPRODUCE FIGURE 7(a)
    if c7_short_non_zero:
        idx = 0
    if c7_long_non_zero:
        idx = 1
    if c7_mix1:
        idx = 3
    if c7_mix2:
        idx = 3
    if c7_c8_mix:
        idx = 4

    input_arr = X_verify[idx]
    print("----- X values -----")
    print(r"$m_\chi = $" + str(X_verify[idx,0]))
    print(r"$c_7^s = $"  + str(X_verify[idx,1]))
    print(r"$c_7^l = $"  + str(X_verify[idx,2]))
    print(r"$c_8^s = $"  + str(X_verify[idx,3]))
    print(r"$c_8^l = $"  + str(X_verify[idx,4]))

    input_arr = input_arr.reshape(1,-1)
    input_arr = normalise_array(input_arr, scaler_quant_X, scaler_stand_X)
    rates_fig_7 = model.predict(input_arr)
    rates_fig_7 = inverse_normalise_array(rates_fig_7, scaler_quant_Y, scaler_stand_Y)

    Qs = np.linspace(1, n_Q_values+1, n_Q_values)
    fig_7_norm = 1/(365.25*1000)


    # Plot the predictions
    plt.step(Qs, Y_verify[idx,:].flatten()*fig_7_norm, label='Data', where='mid')
    # Plot the data
    plt.step(Qs, rates_fig_7.flatten()*fig_7_norm, label='Prediction', where='mid')
    
    plt.title(r'Rates as a function of $e^-$ hole pairs with $m_\chi=$' + str(X_verify[idx,0]) + r' ev, $c_7^s=$' + str(X_verify[idx,1]) + r', c_7^l=' +str(X_verify[idx,2]))
    plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.show()
    """

    """
    # CHeck the R2 score (coefficient of determination)
    val_preds = model.predict(X_val)
    r2 = r2_score(Y_val[:,0], val_preds[:,0])
    print('R2 score for the first Q bin is ' + str(r2))
    """

if __name__ == "__main__":
    main()
