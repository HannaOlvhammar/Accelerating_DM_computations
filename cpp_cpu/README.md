# This directory relies on CPU computations with CUDA. It is used to
# generate dark matter induced transition rates for different numbers of electron hole pairs
# as a function of input parameters such as DM mass and EFT coefficients.

In the first paragraph of generate_data.cpp, you can set the parameter values and ranges.
For example, you can change the detector material and number of rates to generate.

# Usage
Use the makefile with "make" to compile the C++ files. Then run the executable
with "./generate_data".
