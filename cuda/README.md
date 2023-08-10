## GPU accelerated computations for dark matter

This repository relies on GPU computations with CUDA. It is used to
generate dark matter induced transition rates for different numbers of electron hole pairs
as a function of input parameters such as DM mass and EFT coefficients.

In the first paragraph of generate_data.cu, you can set the parameter values and ranges.
For example, you can change the detector material and number of rates to generate.

# Usage
Use the makefile with "make" to compile the CUDA files. Then run the executable
with "./generate_data". Since the GPU computations are performed with CUDA,
you need an Nvidia graphics card to compile the data generation.
