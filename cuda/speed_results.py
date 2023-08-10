import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl



""" In this file, I have documented the time for execution of the different
    implementations of transition rate computations as a function of the
    number of transition rates computed. These are all for the case of having
    n_p = 3 and n_Q = 10.
"""


mpl.style.use('seaborn-v0_8-whitegrid')

# GPU times in seconds
gpu_times = np.array([0*60 + 22.629,      # 1e0 
                      0*60 + 23.943,      # 1e1
                      0*60 + 37.597,      # 1e2
                      1*60 + 5.240,       # 1e3
                      2*60 + 0.660,       # 1e4
                      15*60 + 49.209,     # 1e5
                      9001.16             # 1e6   (150 minutes)
                      ])

# CPU (C++ unparallelised) times in seconds
cpu_times = np.array([0*60 + 10.299,      # 1e0
                      0*60 + 11.288,      # 1e1
                      0*60 + 25.623,      # 1e2,  
                      2*60 + 44.196,      # 1e3,
                      25*60 + 32.832      # 1e4,
                      ])


# Times for the original Python program
py_times = np.array([0*60 + 18.490,         # 1e0
                     0*60 + 19.533,         # 1e1
                     0*60 + 41.645,         # 1e2,  
                     4*60 + 21.288,         # 1e3,
                     38*60 + 34.363         # 1e4,
                    ])


# Neural network times for prediction in seconds
nn_times = np.array([0*60 + 2.805,         # 1e0
                     0*60 + 3.320,         # 1e1
                     0*60 + 3.308,         # 1e2,  
                     0*60 + 3.397,         # 1e3,
                     0*60 + 3.826,         # 1e4,
                     0*60 + 8.560,         # 1e5,
                     0*60 + 56.479         # 1e6,
                    ])


data_sizes_gpu = [1E0, 1E1, 1E2, 1E3, 1E4, 1E5, 1E6]
data_sizes_cpu = [1E0, 1E1, 1E2, 1E3, 1E4]
data_sizes_py = [1E0, 1E1, 1E2, 1E3, 1E4]
data_sizes_nn = [1E0, 1E1, 1E2, 1E3, 1E4, 1E5, 1E6]

plt.plot(data_sizes_gpu, gpu_times, label="GPU C++")
plt.plot(data_sizes_cpu, cpu_times, label="CPU C++")
plt.plot(data_sizes_py, py_times, label="Python")
plt.plot(data_sizes_nn, nn_times, label="ANN")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Number of data points computed")
plt.ylabel("Execution time [s]")
plt.legend()
plt.savefig("../../../report_figures/speed_results.png", dpi=1200)
#plt.show()
