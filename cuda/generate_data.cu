#include <iostream>    // input and output
#include <string>      // string formats 
#include <fstream>     // read/write files
#include <limits>      // type limits
#include <vector>      // vector objects
#include <cmath>        // mathematical tools
#include <cstddef>     // size_t
#include <iomanip>     // precision on outputs
#include <random>      // random uniform distribution
#include "units.h"
#include "rate_calculator.h"

int main()
{
  ///////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////// SET PARAMETERS HERE //////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  // Number of electron hole pairs
  std::size_t n_Q = 10;
  // Number of data points to generate
  std::size_t data_size = 1e6;
  // Determine the crystal element
  bool element_Si = true;
  bool element_Ge = false;
  // Set mass parameter ranges
  double mass_min = 1e6;
  double mass_max = 1e9;
  // Set EFT coefficient ranges
  double c_min = 0;
  double c_max = 1.;
  
  /* -------------------------------- WARNING -------------------------------------------
   * If you change the number of parameters, check:
   * 1. That n_params is correct
   * 2. That the correct parameters are non-zero in the generation of parameters
   * 3. That the new parameters are printed to the output file
   * (Only need to alter this file) 
   * ------------------------------------------------------------------------------------
   */

  // Number of input parameters for calculating the rate 
  std::size_t n_params = 1 + 2;

  // DM spin
  double j_chi = 0.5;

  double rho = 0.4*units::GeV/(units::cm*units::cm*units::cm);
  double median_v0 = 220*units::km/units::sec;
  double median_v_E = 244*units::km/units::sec;
  double median_v_esc = 544*units::km/units::sec;


  ///////////////////////////////////////////////////////////////////////////////////////
  //////// Read the files with momenta, energies and crystal response functions /////////
  ///////////////////////////////////////////////////////////////////////////////////////

  // Open the files depending on crystal element
  std::ifstream q_E_file;
  std::ifstream W_file;
  if (element_Si) { 
    q_E_file.open("../Si106_3/q_1d.dat"); 
    W_file.open("../Si106_3/W_1d.dat"); 
  }
  if (element_Ge) { 
    q_E_file.open("../Ge116_100/q_1d.dat");
    W_file.open("../Ge116_100/W_1d.dat");
  }

  // Check for errors in opening the files
  if (q_E_file.fail()) {
    std::cout << "Error opening q data file" << std::endl;
    return -1;
  }
  if (W_file.fail()) {
    std::cout << "Error opening W data file" << std::endl;
    return -1;
  }
  // Get the q values and E values from the data file
  std::size_t file_size = 4000000; // TODO: Automate reading of file size
  
  std::vector<double> q_read(file_size);
  std::vector<double> E_read(file_size);
  for (std::size_t i = 0; i < file_size; i++) {
    q_E_file >> q_read[i] >> E_read[i];
  }
  q_E_file.close();


  // Get maximum and minimum q and E
  std::size_t vec_size = std::sqrt(file_size);
  double max_q = std::numeric_limits<double>::min();
  double min_q = std::numeric_limits<double>::max();
  double max_E = std::numeric_limits<double>::min();
  double min_E = std::numeric_limits<double>::max();
  for (std::size_t i = 0; i < file_size; i++) { 
    double q_el = q_read[i];
    double E_el = E_read[i];
    if      (q_el > max_q) { max_q = q_el; }
    else if (q_el < min_q) { min_q = q_el; }
    if      (E_el > max_E) { max_E = E_el; }
    else if (E_el < min_E) { min_E = E_el; }
  }

  double dq = (max_q - min_q) / (vec_size-1);
  double dE = (max_E - min_E) / (vec_size-1);
  
  std::vector<double> q_vec(vec_size);
  std::vector<double> E_vec(vec_size);
  for (std::size_t i = 0; i < vec_size; i++) {
    q_vec[i] = min_q + i*dq;
    E_vec[i] = min_E + i*dE;
  }


  std::vector<double> W1(file_size);
  std::vector<double> ReW2(file_size);
  std::vector<double> ImW2(file_size);
  std::vector<double> W3(file_size);
  std::vector<double> W4(file_size);
  std::vector<double> W5(file_size);
  for (std::size_t i = 0; i < vec_size*vec_size; i++) {
    W_file >> W1[i] >> ReW2[i] >> ImW2[i] >> W3[i] >> W4[i] >> W5[i];
  }
  W_file.close();
  // Note that at the moment, Ws are called with W[vec_size*row + column],
  // which is equivalent to W[row][column]


  ///////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////// Generate the input parameters ////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  
  std::uniform_real_distribution<> c_distribution(0., 1.);
  std::uniform_real_distribution<> mass_distribution(6., 9.);
  
  // Generate a logarithmically uniform distribution of masses
  std::vector<double> m_chis(data_size);
  double exponent_min = log10(mass_min);
  double exponent_max = log10(mass_max);
  double step_size = (exponent_max - exponent_min) / (data_size-1);
  for (std::size_t i = 0; i < data_size; i++) {
    m_chis[i] = pow(10, mass_distribution(gen));
  }
  
  // Allocate both the long and short range EFT coefficients for the 14 observables
  std::size_t n_coeffs = 2*14;
  std::vector<double> cvecs(data_size*n_coeffs);
  std::size_t n_cols_cvecs = n_coeffs;
  double c_step_size = (c_max - c_min) / (data_size-1);
  for (std::size_t i = 0; i < data_size; i++) {
    for (std::size_t c = 0; c < n_coeffs; c++) {
      // Set everything to zero as a starting point
      cvecs[i*n_cols_cvecs+c] = 0.;
    }
    // Choose non-zero parameters
    cvecs[i*n_cols_cvecs+5]  = c_distribution(gen);   // c_7 short
    cvecs[i*n_cols_cvecs+19] = c_distribution(gen);   // c_7 long
  }


  // Fix the velocity parameters for now
  double v0s    = median_v0;
  double v_Es   = median_v_E;
  double v_escs = median_v_esc;


  ///////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////   CUDA PART   ////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
 
  // Allocate memory on the GPU
  double* d_rates;
  double* d_cvecs;
  double* d_m_chis;
  double* d_E_vec;
  double* d_q_vec;
  double* d_W1;
  double* d_ReW2;
  double* d_ImW2;
  double* d_W3;
  double* d_W4;
  double* d_W5;

  cudaMalloc(&d_rates,  data_size*n_Q     *sizeof(double));
  cudaMalloc(&d_cvecs,  data_size*n_coeffs*sizeof(double));
  cudaMalloc(&d_m_chis, data_size         *sizeof(double));
  cudaMalloc(&d_E_vec,  vec_size          *sizeof(double));
  cudaMalloc(&d_q_vec,  vec_size          *sizeof(double));
  cudaMalloc(&d_W1,     vec_size*vec_size *sizeof(double));
  cudaMalloc(&d_ReW2,   vec_size*vec_size *sizeof(double));
  cudaMalloc(&d_ImW2,   vec_size*vec_size *sizeof(double));
  cudaMalloc(&d_W3,     vec_size*vec_size *sizeof(double));
  cudaMalloc(&d_W4,     vec_size*vec_size *sizeof(double));
  cudaMalloc(&d_W5,     vec_size*vec_size *sizeof(double));

  // Initialise the device arrays to the host arrays (we don't need to initialise rates
  // and parameters, since they are computed on the GPU)
  cudaMemcpy(d_cvecs,  &cvecs[0],  data_size*n_coeffs*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_m_chis, &m_chis[0], data_size         *sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_E_vec,  &E_vec[0],  vec_size          *sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_q_vec,  &q_vec[0],  vec_size          *sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_W1,     &W1[0],     vec_size*vec_size *sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ReW2,   &ReW2[0],   vec_size*vec_size *sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ImW2,   &ImW2[0],   vec_size*vec_size *sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_W3,     &W3[0],     vec_size*vec_size *sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_W4,     &W4[0],     vec_size*vec_size *sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_W5,     &W5[0],     vec_size*vec_size *sizeof(double), cudaMemcpyHostToDevice);


  
  ///////////////////////////////////////////////////////////////////////////////////////
  /////////////////////// Calculate the rates for the given parameters //////////////////
  ///////////////////////////////////////////////////////////////////////////////////////

  int block_size = 256;
  int n_blocks = (data_size + block_size - 1) / block_size; 

  // Calculate the rates
  get_rates_and_params<<<n_blocks, block_size>>>(
    element_Ge, element_Si,                  // Determines material
    data_size, n_params, d_rates,            // For saving data
    n_Q, n_coeffs, d_cvecs, d_m_chis,        // Input parameters
    mass_min, mass_max, c_min, c_max,        // Input parameter ranges
    v0s, v_Es, v_escs, j_chi, rho,           // System units
    d_E_vec, d_q_vec, vec_size, dE, dq,      // Energy and momenta info
    d_W1, d_ReW2, d_ImW2, d_W3, d_W4, d_W5   // Crystal response functions
  );

  // Wait until GPU is done before accessing data
  int status = cudaDeviceSynchronize();
  if (status != 0) {
    std::cout << "GPU computation failed" << std::endl;
    return -1;
  }

  // Copy the GPU computed arrays to the CPU
  std::vector<double> rates(data_size*n_Q);
  std::vector<double> params(data_size*n_params);
  cudaMemcpy(&rates[0],  d_rates,  data_size*n_Q*sizeof(double),      cudaMemcpyDeviceToHost);

  // Free the GPU memory
  cudaFree(d_cvecs);
  cudaFree(d_m_chis);
  cudaFree(d_E_vec);
  cudaFree(d_q_vec);
  cudaFree(d_W1);
  cudaFree(d_ReW2);
  cudaFree(d_ImW2);
  cudaFree(d_W3);
  cudaFree(d_W4);
  cudaFree(d_W5);
  cudaFree(d_rates);


  ///////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////// Save the rates and parameters in files /////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////

  // Open the two files
  std::ofstream rate_file;
  std::ofstream param_file;
  rate_file.open("rates.dat");
  param_file.open("parameters.dat");

  // Write to the file
  for (std::size_t i = 0; i < data_size; i++) {
    param_file << std::setprecision(20) << std::fixed 
               << m_chis[i] << "\t"
               << cvecs[i*n_cols_cvecs + 5] << "\t"
               << cvecs[i*n_cols_cvecs + 19]
               << std::endl;
    for (std::size_t j = 0; j < n_Q; j++) {
      rate_file << std::setprecision(20) << std::fixed 
                << rates[i*n_Q + j] << "\t";
    }
    rate_file << std::endl;
  }

  rate_file.close();
  param_file.close();

  return 0; 
}

