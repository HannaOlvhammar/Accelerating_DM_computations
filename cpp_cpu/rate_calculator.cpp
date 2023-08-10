#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <cstddef>
#include <numbers>
#include "units.h"
#include "rate_calculator.h"

// The velocity integral
double etav0(double v_min, double v0, double v_E, double v_esc)
{
	double etav0;
	if (v_min < v_esc-v_E) {
		etav0 = std::erf((v_E-v_min)/v0) + std::erf((v_E+v_min)/v0) - 
		        4*v_E/(sqrt(std::numbers::pi)*v0) * exp(-v_esc*v_esc/(v0*v0));
	}
	else {
		etav0 = std::erf((v_E-v_min)/v0) + std::erf(v_esc/v0) -
		        2*(v_esc+v_E-v_min)/(std::sqrt(std::numbers::pi)*v0) * std::exp(-v_esc*v_esc/(v0*v0));
	}
	
	return etav0;
}


// The velocity integral
double etav2(double v_min, double v0, double v_E, double v_esc)
{
	double etav2;
	double one_over_v0_sqrd = 1/(v0*v0);
	if (v_min < v_esc - v_E) {
		etav2 = (
			6*std::exp(-(v_min+v_E)*(v_min+v_E)*one_over_v0_sqrd) * 
			(v_E-v_min+std::exp(4*v_min*v_E*one_over_v0_sqrd) * 
			(v_min+v_E))*v0*v0 - 8*std::exp(-v_esc*v_esc*one_over_v0_sqrd) * 
			v_E*(3*v_esc*v_esc+v_E*v_E + 3*v0*v0) + 3*std::sqrt(std::numbers::pi)*v0*(2*v_E*v_E+v0*v0) *
			(std::erf((v_min+v_E)/v0) - std::erf((v_min-v_E)/v0))
			)
			/ (6*std::sqrt(std::numbers::pi)*v0);
  }
	else {
		etav2 = (
			4*std::exp(-v_esc*v_esc*one_over_v0_sqrd)*(v_min*v_min*v_min - 
			(v_esc+v_E)*(v_esc+v_E)*(v_esc+v_E)) + 
			3*v0*(2*std::exp(-(v_min-v_E)*(v_min-v_E)*one_over_v0_sqrd) * (v_min+v_E)*v0 - 
			2*std::exp(-v_esc*v_esc*one_over_v0_sqrd)*(v_esc+2*v_E)*v0 + 
			std::sqrt(std::numbers::pi)*(2*v_E*v_E+v0*v0) *
			(std::erf(v_esc/v0) + std::erf((v_E-v_min)/v0)))
			)
			/ (6*std::sqrt(std::numbers::pi)*v0);
  }

  return etav2;

}


static inline
void get_dm_responses(double* dm_responses,
                      double q,
                      double E,
                      double* cvec,
                      double j_chi,
                      double mu,
                      double m_chi) 
{
  double m_e = units::electronmass; // units
  double qref_over_q_sqrd = m_e*m_e/(q*q*137*137);

	double c1  = cvec[0]   + cvec[14] * qref_over_q_sqrd;
	double c3  = cvec[1]   + cvec[15] * qref_over_q_sqrd;
	double c4  = cvec[2]   + cvec[16] * qref_over_q_sqrd;
	double c5  = cvec[3]   + cvec[17] * qref_over_q_sqrd;
	double c6  = cvec[4]   + cvec[18] * qref_over_q_sqrd;
	double c7  = cvec[5]   + cvec[19] * qref_over_q_sqrd;
	double c8  = cvec[6]   + cvec[20] * qref_over_q_sqrd;
	double c9  = cvec[7]   + cvec[21] * qref_over_q_sqrd;
	double c10 = cvec[8]   + cvec[22] * qref_over_q_sqrd;
	double c11 = cvec[9]   + cvec[23] * qref_over_q_sqrd;
	double c12 = cvec[10]  + cvec[24] * qref_over_q_sqrd;
	double c13 = cvec[11]  + cvec[25] * qref_over_q_sqrd;
	double c14 = cvec[12]  + cvec[26] * qref_over_q_sqrd;
	double c15 = cvec[13]  + cvec[27] * qref_over_q_sqrd;

  // This contains the part of R1 independent of v.
  double q_over_me_sqrd = q*q/(m_e*m_e);
	double q_dot_v_perp = E/m_e - q_over_me_sqrd/2;
	double v_perp_squared = q*q/(4*mu*mu) * (m_chi-m_e)/(m_chi+m_e) - E/mu;
  double q_cross_v_perp_squared = q_over_me_sqrd*v_perp_squared - q_dot_v_perp*q_dot_v_perp;
	
	double R1v0 =	c1*c1 + (c3*c3/4)*q_cross_v_perp_squared + c7*c7/4 * v_perp_squared +
                c10*c10/4 * q_over_me_sqrd + 
                j_chi*(j_chi+1)/12 * (
                3*c4*c4 + (4*c5*c5 - 2*c12*c15) * q_cross_v_perp_squared + 
                c6*c6 * q_over_me_sqrd*q_over_me_sqrd + 
                (4*c8*c8 + 2*c12*c12) * v_perp_squared + 
                (2*c9*c9 + 4*c11*c11 + 2*c4*c6) * q_over_me_sqrd + 
                (c13*c13 + c14*c14) * q_over_me_sqrd * v_perp_squared +
                c15*c15 * q_over_me_sqrd * q_cross_v_perp_squared + 
                c13*c14*q_dot_v_perp*q_dot_v_perp
                );


  //This contains the part of R1 proportional to v^2.
  double R1v2 = c3*c3/4 * q_over_me_sqrd + c7*c7/4 +
                j_chi*(j_chi+1)/12 * ( 
                (4*c5*c5 - 2*c12*c15) * q_over_me_sqrd + (4*c8*c8 + 2*c12*c12) + 
                (c13*c13 + c14*c14) * q_over_me_sqrd +
                c15*c15 * q_over_me_sqrd*q_over_me_sqrd
                );


  // This contains the real part of R2.
  double ReR2 = (E*m_e/(q*q)-0.5) * (-c7*c7/2 - 
	              j_chi*(j_chi+1)/6 * (
	              4*c8*c8 + 2*c12*c12 + (c13+c14)*(c13+c14) * q_over_me_sqrd
                ));

  // This contains the immaginary part of R2.
  double ImR2 = c7*c10/2 + 
	               j_chi*(j_chi+1)/6 * (
	               -c4*c13 - c4*c14 + 2*c9*c12 + 4*c11*c8 -
	               (c6*c13 + c6*c14) * q_over_me_sqrd
	               );


  // This contains all of R3.
  double R3 = c3*c3/4 * q_over_me_sqrd + c7*c7/4 + 
              j_chi*(j_chi+1)/12* ( 
              (4*c5*c5 + c13*c13 + c14*c14 - 2*c12*c15 ) * q_over_me_sqrd + 4*c8*c8 +
              2*c12*c12 + c15*c15 * q_over_me_sqrd*q_over_me_sqrd
              );


  // This contains all of R4.
  double R4 = -c3*c3/4 + 
              j_chi*(j_chi+1)/12 * (
              -4*c5*c5 - c15*c15 * q_over_me_sqrd + 2*c12*c15 + 2*c13*c14
              );


  // This contains all of R5.
  double R5 = j_chi*(j_chi+1)/6 * ( 
	            4*c3*c7 + 4*c5*c8 - c12*c13 + c12*c14 - 4*c14*c15*q_over_me_sqrd
	            );


  dm_responses[0] = R1v0;
  dm_responses[1] = R1v2;
  dm_responses[2] = ReR2;
  dm_responses[3] = ImR2;
  dm_responses[4] = R3;
  dm_responses[5] = R4;
  dm_responses[6] = R5;
}




void get_rates_and_params(bool element_Ge,
                          bool element_Si,
                          std::size_t data_size,
                          std::size_t n_params,
                          double* Qvecs,
                          std::size_t n_Q,
                          double* cvecs,
                          double* m_chis,
                          double v0,
                          double v_E,
                          double v_esc,
                          double j_chi,
                          double rho,
                          double* E_vec,
                          double* q_vec,
                          std::size_t vec_size,
                          double dE,
                          double dq,
                          double* W1,
                          double* ReW2,
                          double* ImW2,
                          double* W3,
                          double* W4,
                          double* W5)
{
	double m_e = units::electronmass;

  // Set the minimum enery that is specific to the material
	double Emin;
	if (element_Ge) { Emin = 0.67; }
	if (element_Si) { Emin = 1.2; }
    
  // Allocate heap memory for the seven DM response function terms
  double R[7];
	

  /* The transition rates will be calculated for different parameter values, i.e.
   * different DM masses and EFT constants. The first iteration will be over then number
   * of data points that was specified by the user in generate_data.c, where each
   * data point corresponds to different parameter values. */

  // Iterate over all data points
  for (size_t iD = 0; iD < data_size; iD++) {
    // Set the corresponding parameter values
    double m_chi = m_chis[iD];
    double mu = m_e*m_chi/(m_e+m_chi);
  
    // Prepare the constants and differentials that will constitute the Riemann sum later
	  double const_and_differentials = rho*dq*dE*units::kg*units::yr
                                    / (m_chi*128*std::numbers::pi*m_chi*m_chi*m_e*m_e)
		                                / (2*(std::erf(v_esc/v0)-2*(v_esc/v0)
                                        *std::exp(-v_esc*v_esc/(v0*v0))
                                        /std::sqrt(std::numbers::pi))*v_E
                                      );
    if (element_Ge) { const_and_differentials /= 135.33e9; }
	  if (element_Si) { const_and_differentials /= 52.33e9; }
 

    /* The integral gets calculated below. It is computed as a Riemann sum by first
     * determining the energy, specified with iE, then the momentum, determined with iq.
     * The lower boundary for the integral is checked with vmin, and the number of
     * electron hole pairs is determined by the detector material. */
    
    // Begin by iterating over energies
    for (size_t iE = 0; iE < vec_size; iE++) {
      double E = E_vec[iE];
      
      // Compute the number of electron hole pairs allowed by the transition energy
      size_t iQ;
      if (element_Ge) { iQ = (std::size_t)std::floor((E-Emin)/3.0); }
      if (element_Si) { iQ = (std::size_t)std::floor((E-Emin)/3.8); }
      // Make sure to only compute the transition rates for the allowed number of
      // electron hole pairs specified by the user in generate_data.c, and the
      // transition energies above the allowed minimum energy for the crystals.
      if (iQ < n_Q  &&  E > Emin) {
        // Next, iterate over momenta
        for (std::size_t iq = 0; iq < vec_size; iq++) {
          double q = q_vec[iq];
          // The integral includes a lower bound for the DM particle velocity determined
          // by the energy, momentum and DM mass
          double vmin = E/q + q/(2*m_chi);
          if (vmin < v_esc + v_E) {
            // For the allowed velocity range, we can compute the velocity integrals
            // given by the eta operator. We compute them in two steps.
            double eta0_el = etav0(vmin, v0, v_E, v_esc) * q/E;
            double etav2_el = etav2(vmin, v0, v_E, v_esc);
            // Now, we compute the DM response functions.
            get_dm_responses(R, q, E, &cvecs[iD*28], j_chi, mu, m_chi);
            double R1v0 = R[0];
            double R1v2 = R[1];
            double ReR2 = R[2];
            double ImR2 = R[3];
            double R3   = R[4];
            double R4   = R[5];
            double R5   = R[6];
            // Next, we get the crystal response functions from the row major matrices
            // that were loaded by the user in generate_data.c and multiply them
            // by the DM response functions.
            std::size_t W_idx = vec_size*iq + iE;
            double term1 = W1[W_idx]   * (R1v0*eta0_el + R1v2*etav2_el*q/E);
            double term2 = ReW2[W_idx] * ReR2*eta0_el;
            double term3 = ImW2[W_idx] * ImR2*eta0_el;
            double term4 = W3[W_idx]   * R3*eta0_el;
            double term5 = W4[W_idx]   * R4*eta0_el;
            double term6 = W5[W_idx]   * R5*eta0_el;
            // Finally, the integrand is given by the sum of the above terms. The
            // Riemann sum is then given by adding this integrand for each iteration
            // over energy and momenta. The integral is computed separately
            // for different numbers of electron hole pairs.
            Qvecs[iD*n_Q + iQ] += term1 + term2 + term3 + term4 + term5 + term6;
          }
        }
      }
    }

    // Finally, the above Riemann sum is missing the differentials dE and dq; multiply
    // the sums by the differentials and the associated constants here. 
    for (std::size_t iQ = 0; iQ < n_Q; iQ++) {
      Qvecs[iD*n_Q + iQ] *= const_and_differentials; 
    }
    // Store the parameters used for generating these particular integrals
  }

  /* The function ends here. It returns void, but has altered the rate arrays and
     parameter arrays to contain the data. */
}
