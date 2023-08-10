import numpy as np
from scipy.stats import loguniform
import units as u
import rate_calculator as r
import time as t
#####################################
####### Set Input parameters ########
#####################################
#t0=t.time()

#Set DM halo parameters around which parameters will be drawn
rho=0.4*u.GeV/u.cm**3
median_v0=220*u.km/u.sec
median_v_E=244*u.km/u.sec
median_v_esc=544*u.km/u.sec
#DM spin
j_x=0.5
# Number of Q-values the network should train on. This number should probably be cranked up?
# Higher n_Q means longer time per training point, and probably also a need for more training points.
n_Q=10
#Number of data points per run. 4E4 takes 2-3 hours on a single core on my laptop. Depending on the number of cores, you can run multiple calculations at the same time.
#data_size=int(4E4)
data_size = int(1E4) #hundred thousand
n_params = 1 + 2 # mass + c_7 long + c_7 short 

Qdata=np.zeros((data_size,n_Q))
params=np.zeros((data_size, n_params))
#Loading W and reshape it
element = 'Si'
if element=='Si':
    q_data=np.loadtxt('../Si106_3/q_1d.dat')
    W_data=np.loadtxt('../Si106_3/W_1d.dat')
if element=='Ge':
    q_data=np.loadtxt('../Ge116_100/q_1d.dat')
    W_data=np.loadtxt('../Ge116_100/W_1d.dat')
n=int(np.sqrt(len(W_data[:,0])))
dq=(np.max(q_data[:,0])-np.min(q_data[:,0]))/(n-1)
dE=(np.max(q_data[:,1])-np.min(q_data[:,1]))/(n-1)
q_vec=np.linspace(np.min(q_data[:,0]),np.max(q_data[:,0]),n)
E_vec=np.linspace(np.min(q_data[:,1]),np.max(q_data[:,1]),n)
W1=np.reshape(W_data[:,0],(n,n))
ReW2=np.reshape(W_data[:,1],(n,n))
ImW2=np.reshape(W_data[:,2],(n,n))
W3=np.reshape(W_data[:,3],(n,n))
W4=np.reshape(W_data[:,4],(n,n))
W5=np.reshape(W_data[:,5],(n,n))
#print('Loading data took '+str(t.time()-t0)+' s')
#t0=t.time()

#m_min = np.log10(5*10**5)
#m_xs = np.random.uniform(low=5E5, high=1E9, size=data_size)
#m_xs = np.logspace(6, 9, data_size)
m_xs = loguniform.rvs(1E6, 1E9, size=data_size)
v0 = median_v0
v_E = median_v_E
v_esc = median_v_esc
cvecs = np.zeros((data_size,28))
#cvecs = np.random.uniform(low=0, high=1, size=(data_size,28))
#cvecs = loguniform.rvs(10**(-10), 1, size=(data_size,28))

# Set only c_7 short and long range to non-zero; test the nn by extracting rate plots
# for c_7 long = 1 and then c_7 short = 1
#cvecs[:,5]  = loguniform.rvs(10**(-5), 1, size=(data_size))  # set only c_7_short \neq 0 (c_2 does not exist)
#cvecs[:,19] = loguniform.rvs(10**(-5), 1, size=(data_size))  # set only c_7_long \neq 0 
#cvecs[:,5]  = np.linspace(0, 1, data_size)  # set only c_7_short \neq 0 (c_2 does not exist)
#cvecs[:,19]  = np.linspace(0, 1, data_size)  # set only c_7_short \neq 0 (c_2 does not exist)
cvecs[:,5]  = np.random.uniform(0, 1, size=(data_size))  # set only c_7_short \neq 0 (c_2 does not exist)
cvecs[:,19] = np.random.uniform(0, 1, size=(data_size))  # set only c_7_long \neq 0 
# Set this to plot worthy values, but don't let NN train on them

# Short is non-zero
#cvecs[0,5] = 1.
#cvecs[0,19] = 0.
# Long is non-zero
#cvecs[1,5] = 0.
#cvecs[1,19] = 1.


#v0s = np.random.uniform(low=median_v0/2.0, high=median_v0*1.5, size=data_size)
#v_Es = np.random.uniform(low=median_v_E/2.0, high=median_v_E*1.5, size=data_size)
#v_escs = np.random.uniform(low=median_v_esc/2.0, high=median_v_esc*1.5, size=data_size)
#cvecs = np.random.uniform(low=0, high=1, size=(data_size,28))
#print('Generating random numbers took '+str(t.time()-t0)+' s')


#t0=t.time()
for i in range(data_size):
    #t0=t.time()
    #Calculate data
    Qdata[i,:] = r.myintegrator(cvecs[i,:],m_xs[i],v0,v_E,v_esc,j_x,rho,element,n_Q,E_vec,q_vec,dE,dq,W1,ReW2,ImW2,W3,W4,W5)
    #Store parameters used to generate the data. Rescaling parameters to be between 0 and 1. 
    #params[i] = m_xs[i]/(1E9)
    params[i,0] = m_xs[i]
    params[i,1] = cvecs[i,5]
    params[i,2] = cvecs[i,19]
    #print('Computing rate ' + str(i) + ' took ' + str(t.time()-t0))
    #print(Qdata[i,:])
#    if ((i+1) % 100 == 0):
#        print(str(i+1) + " rates computed")
#print('Computing ' + str(data_size) + ' rates took '+ str((t.time()-t0)/60.0) + ' min')

#Save the data and parameters.
np.save('Qs.npy',Qdata)
np.save('params.npy',params)
