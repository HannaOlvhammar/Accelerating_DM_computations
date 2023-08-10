import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
# Disable tensorflow info, warning and error messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

#########################################################################################
###################################  Preprocess data  ###################################
#########################################################################################


def normalise_array(data, scaler_quant, scaler_stand):
    data = scaler_quant.transform(data)
    data = scaler_stand.transform(data)
    return data

def inverse_normalise_array(data, scaler_quant, scaler_stand):
    data = scaler_stand.inverse_transform(data)
    data = scaler_quant.inverse_transform(data)
    return data

def normalise_set(train, val):
    scaler_quant  = QuantileTransformer().fit(train)
    train         = scaler_quant.transform(train)
    val           = scaler_quant.transform(val)
    #plt.hist(train, bins=25)
    #plt.ylabel("Frequency")
    #plt.xlabel(r"$m_\chi$ after quantiles transformation")
    #plt.savefig("../../../report_figures/masses_after_quantiles.png", dpi=1200)
    #plt.show()
    scaler_stand  = StandardScaler().fit(train)
    train         = scaler_stand.transform(train)
    val           = scaler_stand.transform(val)
    return train, val, scaler_quant, scaler_stand

def inverse_normalise_set(train, val, scaler_quant, scaler_stand):
    train = scaler_stand.inverse_transform(train)
    val   = scaler_stand.inverse_transform(val)
    train = scaler_quant.inverse_transform(train)
    val   = scaler_quant.inverse_transform(val)
    return train, val



def main():

    mpl.style.use('seaborn-v0_8-whitegrid')
    X_verify = np.load('../check_data/params.npy')
    Y_verify = np.load('../check_data/rates.npy')
    

    #########################################################################################
    ###########################  Load and normalise data data  ##############################
    #########################################################################################

    element = 'Si'
    #element = 'Ge'


    # Load data; data needs to be generated in separate file
    if element == 'Si':
        X = np.loadtxt('../data_dirs/1_param_1e5_points/parameters.dat')
        Y = np.loadtxt('../data_dirs/1_param_1e5_points/rates.dat')
    #if element == 'Ge':
    #    X = np.load('ge_m_data/params.npy')
    #    Y = np.load('ge_m_data/Qs.npy')
    
    # Because X has only masses
    X = X.reshape(-1,1)

    # Split features and labels into training and validation sets. Shuffle
    # to ensure that the NN trains on all kinds of generated values
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.33, shuffle=True)

    X_train, X_val, scaler_quant_X, scaler_stand_X = normalise_set(X_train, X_val)
    Y_train, Y_val, scaler_quant_Y, scaler_stand_Y = normalise_set(Y_train, Y_val)
    
    # Record properties of the data
    train_data_size = np.shape(X_train)[0]
    val_data_size = np.shape(X_val)[0]
    n_data_points = train_data_size + val_data_size
    n_features = np.shape(X_train)[1]
    n_Q_values = np.shape(Y_train[1])[0]


    #""" VERIFY: Masses logarithmically uniformly distributed
    #----------------------------------------------------------------------------------------
    plt.hist(X, bins=np.logspace(np.log10(np.min(X)), np.log10(np.max(X)), 50))
    plt.xscale("log")
    plt.ylabel("Frequency")
    plt.xlabel(r"$m_\chi$ [eV]")
    plt.show()
    #----------------------------------------------------------------------------------------
    #"""


    #""" VERIFY: Masses nicely scaled
    #----------------------------------------------------------------------------------------
    plt.hist(X_train, bins=25)
    plt.ylabel("Frequency")
    plt.xlabel(r"$m_\chi$ after quantiles and standard transformation")
    #plt.savefig("../../../report_figures/masses_after_standard.png")
    plt.show()
    #----------------------------------------------------------------------------------------
    #"""


    #""" VERIFY: Original distribution of the rates
    #----------------------------------------------------------------------------------------
    plt.hist(Y, bins=np.logspace(-1, 7, 50))
    plt.xscale("log")
    plt.ylabel("Frequency")
    plt.xlabel(r"$m_\chi$ [eV]")
    plt.show()
    #----------------------------------------------------------------------------------------
    #"""


    #""" VERIFY: Rates have been scaled in a reasonable way (mean 0, variance 0)
    #-------------------------------------------------------------------------------------
    plt.hist(Y_train) 
    plt.title('Generated rates, scaled')
    plt.show()
    #-------------------------------------------------------------------------------------
    #"""


    #""" VERIFY: Shapes of the data
    #-------------------------------------------------------------------------------------
    # Check shapes of the data
    print("Shape of training input: \t" + str(np.shape(X_train)))
    print("Shape of training output: \t" + str(np.shape(Y_train)))
    print("Shape of validation input: \t" + str(np.shape(X_val)))
    print("Shape of validation input: \t" + str(np.shape(Y_val)))
    #-------------------------------------------------------------------------------------
    #"""


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

    # Save the model
    model.save('model_mass')

    test_loss, test_acc = model.evaluate(X_val,  Y_val)
    predictions = model.predict(X_val)


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
    plt.xlabel('Epochs')
    plt.grid()
    plt.savefig("../../../report_figures/loss_function_mass.png")
    plt.show()

    # Check if there is a connection between bad predictions and mass size
    X_sort_inds = np.argsort(X_val[:,0], axis=0).flatten() # sort by masses
    X_val_sorted = X_val[X_sort_inds,:]
    Y_val_sorted = Y_val[X_sort_inds,:]
    predictions_sorted = predictions[X_sort_inds,:]


    for i in range(n_Q_values):
        plt.plot(X_val_sorted[:,0], Y_val_sorted[:,i]/(365.25*1000), 'o', alpha=0.5, label='Validation Data')
        plt.plot(X_val_sorted[:,0], predictions_sorted[:,i]/(365.25*1000), '-', label='Prediction')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid()
        plt.ylabel('Transition rate [1 / g day]')
        plt.xlabel(r'$m_\chi$ [eV]')
        plt.title(r'$Q = $' + str(i+1)) #+ " for " + element)
        plt.legend()
        plt.savefig("../../../report_figures/1e5_rate_for_mass_Q" + str(i+1) + ".png", dpi=1200)
        plt.show()


    # REPRODUCE FIGURE 7(a)
    mass_10_MeV = np.array(10**7).reshape(-1,1)
    #mass_10_MeV = np.array(10**7, 1, 0).reshape(-1,1)
    mass_10_MeV = normalise_array(mass_10_MeV, scaler_quant_X, scaler_stand_X)
    rates_fig_7 = model.predict(mass_10_MeV)
    rates_fig_7 = inverse_normalise_array(rates_fig_7, scaler_quant_Y, scaler_stand_Y)

    """ Only masses
    """
    rate_idx = []
    for i in range(len(X_val)):
        if X_val[i] < 10**7+5000  and  X_val[i]>10**7-5000:
            rate_idx.append(i)
            print(X_val[i])

    plt.step(np.linspace(1, n_Q_values+1, n_Q_values), Y_verify[0].flatten()/(365.25*1000), label='Data', where='mid')
    plt.step(np.linspace(1, n_Q_values+1, n_Q_values), rates_fig_7.flatten()/(365.25*1000), label='Prediction', where='mid')
    plt.yscale('log')
    plt.grid()
    plt.title(r'$m_\chi = 10$ MeV, $c_7^s = 1$')# + element)
    plt.ylabel(r'Transition rate [1 / g day]')
    plt.xlabel(r'$Q$')
    plt.legend()
    plt.savefig("../../../report_figures/1e5_rates_for_Q.png", dpi=1200)
    plt.show()




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



if __name__ == "__main__":
    main()
