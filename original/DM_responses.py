from numba import jit

#This contains the part of R1 independent of v.
@jit
def R1v0(q,E,cvec,j_chi,m_e,mu,m_chi):
	c1=cvec[0]+cvec[14]*m_e**2/(q*137)**2
	c3=cvec[1]+cvec[15]*m_e**2/(q*137)**2
	c4=cvec[2]+cvec[16]*m_e**2/(q*137)**2
	c5=cvec[3]+cvec[17]*m_e**2/(q*137)**2
	c6=cvec[4]+cvec[18]*m_e**2/(q*137)**2
	c7=cvec[5]+cvec[19]*m_e**2/(q*137)**2
	c8=cvec[6]+cvec[20]*m_e**2/(q*137)**2
	c9=cvec[7]+cvec[21]*m_e**2/(q*137)**2
	c10=cvec[8]+cvec[22]*m_e**2/(q*137)**2
	c11=cvec[9]+cvec[23]*m_e**2/(q*137)**2
	c12=cvec[10]+cvec[24]*m_e**2/(q*137)**2
	c13=cvec[11]+cvec[25]*m_e**2/(q*137)**2
	c14=cvec[12]+cvec[26]*m_e**2/(q*137)**2
	c15=cvec[13]+cvec[27]*m_e**2/(q*137)**2
	q_dot_v_perp = E/m_e - q**2/(2*m_e**2)
	v_perp_squared = q**2/(4*mu**2) * (m_chi-m_e)/(m_chi+m_e) - E/mu
	q_cross_v_perp_squared = q**2/m_e**2 * v_perp_squared - q_dot_v_perp**2
	
	result = c1**2 + c3**2/4 * q_cross_v_perp_squared + c7**2/4 * v_perp_squared + \
	         c10**2/4 * q**2/m_e**2 + j_chi*(j_chi+1)/12 * ( 3*c4**2 + \
	         ( 4*c5**2 - 2*c12*c15 ) * q_cross_v_perp_squared + c6**2 * q**4/m_e**4 + \
	         ( 4*c8**2 + 2*c12**2 ) * v_perp_squared + ( 2*c9**2 + 4*c11**2 + 2*c4*c6) *\
	         q**2/m_e**2 + ( c13**2 + c14**2 ) * q**2/m_e**2 * v_perp_squared + \
	         c15**2 * q**2/m_e**2 * q_cross_v_perp_squared + c13*c14*q_dot_v_perp**2)
	
	return result

@jit
def R1v2(q,E,cvec,j_chi,m_e,mu,m_chi):	#This contains the part of R1 proportional to v^2.
	c1=cvec[0]+cvec[14]*m_e**2/(q*137)**2
	c3=cvec[1]+cvec[15]*m_e**2/(q*137)**2
	c4=cvec[2]+cvec[16]*m_e**2/(q*137)**2
	c5=cvec[3]+cvec[17]*m_e**2/(q*137)**2
	c6=cvec[4]+cvec[18]*m_e**2/(q*137)**2
	c7=cvec[5]+cvec[19]*m_e**2/(q*137)**2
	c8=cvec[6]+cvec[20]*m_e**2/(q*137)**2
	c9=cvec[7]+cvec[21]*m_e**2/(q*137)**2
	c10=cvec[8]+cvec[22]*m_e**2/(q*137)**2
	c11=cvec[9]+cvec[23]*m_e**2/(q*137)**2
	c12=cvec[10]+cvec[24]*m_e**2/(q*137)**2
	c13=cvec[11]+cvec[25]*m_e**2/(q*137)**2
	c14=cvec[12]+cvec[26]*m_e**2/(q*137)**2
	c15=cvec[13]+cvec[27]*m_e**2/(q*137)**2
	result = c3**2/4 * q**2/m_e**2 + c7**2/4 + j_chi*(j_chi+1)/12 * ( (4*c5**2 - 2*c12*c15) *\
	         q**2/m_e**2 + ( 4*c8**2 + 2*c12**2 ) + ( c13**2 + c14**2 ) * q**2/m_e**2 + \
	         c15**2 * q**4/m_e**4)
	return result

@jit
def ReR2(q,E,cvec,j_chi,m_e,mu,m_chi):	#This contains the real part of R2.
	c1=cvec[0]+cvec[14]*m_e**2/(q*137)**2
	c3=cvec[1]+cvec[15]*m_e**2/(q*137)**2
	c4=cvec[2]+cvec[16]*m_e**2/(q*137)**2
	c5=cvec[3]+cvec[17]*m_e**2/(q*137)**2
	c6=cvec[4]+cvec[18]*m_e**2/(q*137)**2
	c7=cvec[5]+cvec[19]*m_e**2/(q*137)**2
	c8=cvec[6]+cvec[20]*m_e**2/(q*137)**2
	c9=cvec[7]+cvec[21]*m_e**2/(q*137)**2
	c10=cvec[8]+cvec[22]*m_e**2/(q*137)**2
	c11=cvec[9]+cvec[23]*m_e**2/(q*137)**2
	c12=cvec[10]+cvec[24]*m_e**2/(q*137)**2
	c13=cvec[11]+cvec[25]*m_e**2/(q*137)**2
	c14=cvec[12]+cvec[26]*m_e**2/(q*137)**2
	c15=cvec[13]+cvec[27]*m_e**2/(q*137)**2
	result=(E*m_e/q**2-0.5)*( -c7**2/2 - j_chi*(j_chi+1)/6*(4*c8**2 + 2*c12**2 + (c13+c14)**2 * q**2/m_e**2))
	return result

@jit
def ImR2(q,E,cvec,j_chi,m_e,mu,m_chi):#This contains the immaginary part of R2.
	c1=cvec[0]+cvec[14]*m_e**2/(q*137)**2
	c3=cvec[1]+cvec[15]*m_e**2/(q*137)**2
	c4=cvec[2]+cvec[16]*m_e**2/(q*137)**2
	c5=cvec[3]+cvec[17]*m_e**2/(q*137)**2
	c6=cvec[4]+cvec[18]*m_e**2/(q*137)**2
	c7=cvec[5]+cvec[19]*m_e**2/(q*137)**2
	c8=cvec[6]+cvec[20]*m_e**2/(q*137)**2
	c9=cvec[7]+cvec[21]*m_e**2/(q*137)**2
	c10=cvec[8]+cvec[22]*m_e**2/(q*137)**2
	c11=cvec[9]+cvec[23]*m_e**2/(q*137)**2
	c12=cvec[10]+cvec[24]*m_e**2/(q*137)**2
	c13=cvec[11]+cvec[25]*m_e**2/(q*137)**2
	c14=cvec[12]+cvec[26]*m_e**2/(q*137)**2
	c15=cvec[13]+cvec[27]*m_e**2/(q*137)**2
	result = c7*c10/2 + j_chi*(j_chi+1)/6 * ( -c4*c13 - c4*c14 + 2*c9*c12 + 4*c11*c8 - \
	         (c6*c13 + c6*c14) * q**2/m_e**2 )
	return result
@jit
def R3(q,E,cvec,j_chi,m_e,mu,m_chi):#This contains all of R3.
	c1=cvec[0]+cvec[14]*m_e**2/(q*137)**2
	c3=cvec[1]+cvec[15]*m_e**2/(q*137)**2
	c4=cvec[2]+cvec[16]*m_e**2/(q*137)**2
	c5=cvec[3]+cvec[17]*m_e**2/(q*137)**2
	c6=cvec[4]+cvec[18]*m_e**2/(q*137)**2
	c7=cvec[5]+cvec[19]*m_e**2/(q*137)**2
	c8=cvec[6]+cvec[20]*m_e**2/(q*137)**2
	c9=cvec[7]+cvec[21]*m_e**2/(q*137)**2
	c10=cvec[8]+cvec[22]*m_e**2/(q*137)**2
	c11=cvec[9]+cvec[23]*m_e**2/(q*137)**2
	c12=cvec[10]+cvec[24]*m_e**2/(q*137)**2
	c13=cvec[11]+cvec[25]*m_e**2/(q*137)**2
	c14=cvec[12]+cvec[26]*m_e**2/(q*137)**2
	c15=cvec[13]+cvec[27]*m_e**2/(q*137)**2
	result = c3**2/4 * q**2/m_e**2 + c7**2/4 + j_chi*(j_chi+1)/12*( ( 4*c5**2 + c13**2 +\
	         c14**2 - 2*c12*c15 ) * q**2/m_e**2 + 4*c8**2 + 2*c12**2 + c15**2 *q**4/m_e**4 )
	return result

@jit
def R4(q,E,cvec,j_chi,m_e,mu,m_chi):#This contains all of R4.
	c1=cvec[0]+cvec[14]*m_e**2/(q*137)**2
	c3=cvec[1]+cvec[15]*m_e**2/(q*137)**2
	c4=cvec[2]+cvec[16]*m_e**2/(q*137)**2
	c5=cvec[3]+cvec[17]*m_e**2/(q*137)**2
	c6=cvec[4]+cvec[18]*m_e**2/(q*137)**2
	c7=cvec[5]+cvec[19]*m_e**2/(q*137)**2
	c8=cvec[6]+cvec[20]*m_e**2/(q*137)**2
	c9=cvec[7]+cvec[21]*m_e**2/(q*137)**2
	c10=cvec[8]+cvec[22]*m_e**2/(q*137)**2
	c11=cvec[9]+cvec[23]*m_e**2/(q*137)**2
	c12=cvec[10]+cvec[24]*m_e**2/(q*137)**2
	c13=cvec[11]+cvec[25]*m_e**2/(q*137)**2
	c14=cvec[12]+cvec[26]*m_e**2/(q*137)**2
	c15=cvec[13]+cvec[27]*m_e**2/(q*137)**2
	result = -c3**2/4 + j_chi*(j_chi+1)/12*( -4*c5**2 - c15**2 * q**2/m_e**2 +\
	         2*c12*c15 + 2*c13*c14 )
	return result

@jit
def R5(q,E,cvec,j_chi,m_e,mu,m_chi):#This contains all of R5.
	c1=cvec[0]+cvec[14]*m_e**2/(q*137)**2
	c3=cvec[1]+cvec[15]*m_e**2/(q*137)**2
	c4=cvec[2]+cvec[16]*m_e**2/(q*137)**2
	c5=cvec[3]+cvec[17]*m_e**2/(q*137)**2
	c6=cvec[4]+cvec[18]*m_e**2/(q*137)**2
	c7=cvec[5]+cvec[19]*m_e**2/(q*137)**2
	c8=cvec[6]+cvec[20]*m_e**2/(q*137)**2
	c9=cvec[7]+cvec[21]*m_e**2/(q*137)**2
	c10=cvec[8]+cvec[22]*m_e**2/(q*137)**2
	c11=cvec[9]+cvec[23]*m_e**2/(q*137)**2
	c12=cvec[10]+cvec[24]*m_e**2/(q*137)**2
	c13=cvec[11]+cvec[25]*m_e**2/(q*137)**2
	c14=cvec[12]+cvec[26]*m_e**2/(q*137)**2
	c15=cvec[13]+cvec[27]*m_e**2/(q*137)**2
	result = j_chi*(j_chi+1)/6*( 4*c3*c7 + 4*c5*c8 - c12*c13 + c12*c14 - 4*c14*c15*q**2/m_e**2 )
	return result
