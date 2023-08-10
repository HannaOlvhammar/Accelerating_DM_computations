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

    # Set plotting settings
    mpl.style.use('seaborn-v0_8-whitegrid')


    #########################################################################################
    ################################# Load and normalise data ##############################
    #########################################################################################

    check_normalisation = False

    # The data was generated with the mass, c7_short and c7_long varying. Decide here which
    # c parameters should be plotted in the end; does not affect training
    c7_short_non_zero = True
    c7_long_non_zero  = False
    c7_mix            = False

    
    
    ## GET THE PLOTTING DATA
    #X_verify = np.load('si_c7_m_data/single_params.npy')
    #Y_verify_ = np.load('si_c7_m_data/single_rates.npy')
    #Y_verify = Y_verify_[:,:7]

    # Load data that was generated with generate_data.py in si_c7_m_data 
    X = np.loadtxt('../data_dirs/3_params_1e6_points/parameters.dat')
    Y = np.loadtxt('../data_dirs/3_params_1e6_points/rates.dat')


    # Split features and labels into training and validation sets. Shuffle
    # to ensure that the NN trains on all kinds of generated values
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.33, shuffle=True)

    """
    desired_data_size = 1E6
    train_max = int(0.67*desired_data_size)
    val_max = int(0.33*desired_data_size)
    X_train = X_train[:train_max]
    X_val = X_val[:val_max]
    Y_train = Y_train[:train_max]
    Y_val = Y_val[:val_max]
    print(np.shape(X_train)) 
    """

    if check_normalisation:
        # VERIFY: Masses logarithmically uniformly distributed
        plt.hist(X_train, bins=np.logspace(np.log10(np.min(X_train)), np.log10(np.max(X_train)), 25))
        plt.gca().set_xscale("log")
        #plt.title('Generated mass, NO scaling')
        """
        plt.savefig('../../../report_figures/masses_original.png', dpi=1200)
        """
        plt.show()

    # Transform the loguniform distribution of data into uniform distribution with
    # zero mean and unit variance
    X_train, X_val, scaler_quant_X, scaler_stand_X = normalise_set(X_train, X_val)
    Y_train, Y_val, scaler_quant_Y, scaler_stand_Y = normalise_set(Y_train, Y_val)
    
    # Record properties of the data and print at the end
    train_data_size = np.shape(X_train)[0]
    val_data_size = np.shape(X_val)[0]
    n_data_points = train_data_size + val_data_size
    n_features = np.shape(X_train)[1]
    n_Q_values = np.shape(Y_train[1])[0]


    if check_normalisation:
        # VERIFY: Masses nicely scaled
        plt.hist(X_train, bins=25)
        plt.title('Generated mass, scaled')
        plt.show()

        # VERIFY: Original distribution of the rates
        plt.hist(Y, bins=np.logspace(-1, 7, 25))
        plt.gca().set_xscale("log")
        plt.title('Generated rates, NO scaling')
        plt.show()

        # VERIFY: Rates have been scaled in a reasonable way (mean 0, variance 0)
        plt.hist(Y_train, bins=25) 
        plt.title('Generated rates, scaled')
        plt.show()

        # VERIFY: Shapes of the data
        print("Shape of training input: \t" + str(np.shape(X_train)))
        print("Shape of training output: \t" + str(np.shape(Y_train)))
        print("Shape of validation input: \t" + str(np.shape(X_val)))
        print("Shape of validation input: \t" + str(np.shape(Y_val)))


    #########################################################################################
    ###################################  Train the network  #################################
    #########################################################################################

    # Set up the model layers
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(n_features,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(units=n_Q_values),
    ])

    # Build the model and output the setup
    model.build()
    model.summary()

    # Set up the optimizer
    #learning_rate = 1e-6
    #optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer = tf.keras.optimizers.Adam()

    # Compile the model with the chosen optimizer and loss function
    model.compile(optimizer=optimizer,
                  loss='MSE',
                  metrics=['RootMeanSquaredError'],
                  )

    # Fit the model to the training data
    history = model.fit(X_train,
                        Y_train,
                        epochs=30,
                        validation_data=(X_val, Y_val))


    test_loss, test_acc = model.evaluate(X_val,  Y_val)
    predictions = model.predict(X_val)

    # Save the model
    model.save('model_eft_1e6')
    



    #########################################################################################
    ###################################  Analyse results  ###################################
    #########################################################################################

    # Describe the dimensions of the problem
    print('Size of training data: \t\t' + str(train_data_size) + '\n' +
          'Size of validation data: \t' + str(val_data_size) + '\n' +
          'Number of features: \t\t' + str(n_features) + '\n' +
          'Number of labels: \t\t' + str(n_Q_values) + '\n')

    # Scale back the features, labels and the predictions
    X_train, X_val = inverse_normalise_set(X_train, X_val, scaler_quant_X, scaler_stand_X)
    Y_train, Y_val = inverse_normalise_set(Y_train, Y_val, scaler_quant_Y, scaler_stand_Y)
    predictions    = inverse_normalise_array(predictions, scaler_quant_Y, scaler_stand_Y)

    # PLOT the loss function for both the training and validation data as a function of
    # the number of epochs
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend()
    plt.ylabel('MSE')
    plt.xlabel('Number of epochs')
    plt.grid()
    plt.savefig('../../../report_figures/loss_function.png', dpi=1200)
    plt.show()
    
    # REPRODUCE FIGURE 7(a)
    if c7_short_non_zero:
        idx = 0
    if c7_long_non_zero:
        idx = 1
    if c7_mix:
        idx = 2

    input_arr = X_verify[idx]
    print("----- X values -----")
    print(r"$m_\chi = $" + str(X_verify[idx,0]))
    print(r"$c_7^s = $"  + str(X_verify[idx,1]))
    print(r"$c_7^l = $"  + str(X_verify[idx,2]))

    input_arr = input_arr.reshape(1,-1)
    input_arr = normalise_array(input_arr, scaler_quant_X, scaler_stand_X)
    rates_fig_7 = model.predict(input_arr)
    rates_fig_7 = inverse_normalise_array(rates_fig_7, scaler_quant_Y, scaler_stand_Y)

    Qs = np.linspace(1, n_Q_values+1, n_Q_values)
    fig_7_norm = 1/(365.25*1000)


    # Plot the data
    plt.step(Qs, rates_fig_7.flatten()*fig_7_norm, label='Prediction', where='mid')
    # Plot the predictions
    plt.step(Qs, Y_verify[idx,:].flatten()*fig_7_norm, label='Data', where='mid')
    
    #plt.title(r'$m_\chi=$' + str(X_verify[idx,0]) + r' ev, $c_7^s=$' + str(X_verify[idx,1]) + r', c_7^l=' +str(X_verify[idx,2]))
    plt.title(r'$m_\chi=10$ MeV, $c_7^s=1$, $c_7^l=0$' +str(X_verify[idx,2]))
    plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.savefig("../../../report_figures/rates_Qbins_eft.png", dpi=1200)
    plt.show()


    """
    # CHeck the R2 score (coefficient of determination)
    val_preds = model.predict(X_val)
    r2 = r2_score(Y_val[:,0], val_preds[:,0])
    print('R2 score for the first Q bin is ' + str(r2))
    # Check some values to get a sense of the accuracy
    print('----------------------------------------------------------')
    print('---------------------- Smallest mass ---------------------')
    print('----------------------------------------------------------')
    print('Prediction: \t'+str(predictions_sorted[0,:]))
    print('Data: \t\t'+str(Y_val_sorted[0,:]))
    print('----------------------------------------------------------')
    print('----------------------  Middle mass ----------------------')
    print('----------------------------------------------------------')
    middle_idx = int(len(Y_val_sorted)/2)
    print('Prediction: \t'+str(predictions_sorted[middle_idx,:]))
    print('Data: \t\t'+str(Y_val_sorted[middle_idx,:]))
    print('----------------------------------------------------------')
    print('---------------------- Largest masses --------------------')
    print('----------------------------------------------------------')
    print('Prediction: \t'+str(predictions_sorted[-1,:]))
    print('Data: \t\t'+str(Y_val_sorted[-1,:]))
    print('----------------------------------------------------------')
    print('----------------------------------------------------------')
    print('----------------------------------------------------------')
    """


if __name__ == "__main__":
    main()
