import numpy as np
from scipy.stats import loguniform
import units as u
import rate_calculator as r
import time as t
#####################################
####### Set Input parameters ########
#####################################
t0=t.time()

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
data_size = 5
n_params = 1 + 2# + 2 # mass + c_7 long + c_7 short + c_8 short + c_8 long

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
print('Loading data took '+str(t.time()-t0)+' s')
t0=t.time()

m_xs = np.array([0,0,0,0,0])
m_xs[0] = 1E7 # include this point to be able to plot later
m_xs[1] = 1E7 # include this point to be able to plot later
m_xs[2] = 1E7 # include this point to be able to plot later
m_xs[3] = 1E8 # include this point to be able to plot later
v0 = median_v0
v_E = median_v_E
v_esc = median_v_esc
cvecs = np.zeros((data_size,28))

# Short is non-zero
cvecs[0,5] = 1.
cvecs[0,19] = 0.
# Long is non-zero
cvecs[1,5] = 0.
cvecs[1,19] = 1.
# Mix 1
cvecs[2,5] = 0.5
cvecs[2,19] = 0.5
# Mix 2
cvecs[3,5] = 0.7
cvecs[3,19] = 0.1
#cvecs[3,5] = 0.3
#cvecs[3,19] = 0.6
#cvecs[3,5] = 0.2
#cvecs[3,19] = 0.9

print('Generating random numbers took '+str(t.time()-t0)+' s')

t0=t.time()
for i in range(data_size):
    #t0=t.time()
    #Calculate data
    Qdata[i,:] = r.myintegrator(cvecs[i,:],m_xs[i],v0,v_E,v_esc,j_x,rho,element,n_Q,E_vec,q_vec,dE,dq,W1,ReW2,ImW2,W3,W4,W5)
    #Store parameters used to generate the data. Rescaling parameters to be between 0 and 1. 
    # TODO: Maybe this will need to be changed... why are we scaling here?
    #params[i] = m_xs[i]/(1E9)
    params[i,0] = m_xs[i]
    params[i,1] = cvecs[i,5]
    params[i,2] = cvecs[i,19]
    #params[i,3] = cvecs[i,6]
    #params[i,4] = cvecs[i,20]
    #print('Computing rate ' + str(i) + ' took ' + str(t.time()-t0))
    #print(Qdata[i,:])
    if ((i+1) % 100 == 0):
        print(str(i+1) + " rates computed")
#print('Computing ' + str(data_size) + ' rates took '+ str((t.time()-t0)/60.0) + ' min')

#Save the data and parameters.
np.save('rates.npy',Qdata)
np.save('params.npy',params)
