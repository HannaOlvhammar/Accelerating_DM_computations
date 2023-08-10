#ifndef RATE_CALCULATOR_H
#define RATE_CALCULATOR_H


#include <cstddef>

void get_rates_and_params(bool element_Ge, bool element_Si,
                          std::size_t data_size, std::size_t n_params, double* Qvec,
                          std::size_t n_Q, double* cvecs, double* m_chis,
                          double v0, double v_E, double v_esc, double j_chi, double rho,
                          double* E_vec, double* q_vec, std::size_t vec_size, double dE, double dq,
                          double* W1, double* ReW2, double* ImW2, double* W3, double* W4, double* W5);


#endif
