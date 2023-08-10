import numpy as np
from numba import jit
import math
import units as u
import DM_responses as DM_R
@jit
def etav0(v_min,v0,v_E,v_esc): #The velocity integral
	if v_min<v_esc-v_E:
		return math.erf((v_E-v_min)/v0)+math.erf((v_E+v_min)/v0)-4*v_E/(np.sqrt(np.pi)*v0)*np.exp(-v_esc**2/v0**2)
	else:
		return math.erf((v_E-v_min)/v0) + math.erf(v_esc/v0)-2*(v_esc+v_E-v_min)/(np.sqrt(np.pi)*v0)*np.exp(-v_esc**2/v0**2)
@jit
def etav2(v_min,v0,v_E,v_esc):
	if v_min<v_esc-v_E:
		return (6*np.exp(-(v_min+v_E)**2/v0**2)*(v_E-v_min+np.exp(4*v_min*v_E/v0**2)*(v_min+v_E))*v0**2-8*np.exp(-v_esc**2/v0**2)*v_E*(3*v_esc**2+v_E**2+3*v0**2)+3*np.sqrt(np.pi)*v0*(2*v_E**2+v0**2)*(math.erf((v_min+v_E)/v0)-math.erf((v_min-v_E)/v0)))/(6*np.sqrt(np.pi)*v0)
	else:
		return (4*np.exp(-v_esc**2/v0**2)*(v_min**3-(v_esc+v_E)**3)+3*v0*(2*np.exp(-(v_min-v_E)**2/v0**2)*(v_min+v_E)*v0-2*np.exp(-v_esc**2/v0**2)*(v_esc+2*v_E)*v0+np.sqrt(np.pi)*(2*v_E**2+v0**2)*(math.erf(v_esc/v0)+math.erf((v_E-v_min)/v0))))/(6*np.sqrt(np.pi)*v0)
			

@jit
def myintegrator(cvec,m_chi,v0,v_E,v_esc,j_x,rho,element,n,E_vec,q_vec,dE,dq,W1,ReW2,ImW2,W3,W4,W5):
	mu=u.electronmass*m_chi/(u.electronmass+m_chi)
	if element=='Ge':
		Emin=0.67
	elif element=='Si':
		Emin=1.2
	else:
		print('Error, wrong element name')
	Qvec=np.zeros(n)
	integraltemp=np.zeros(6)
	for iE in range(0,len(E_vec)):
		E=E_vec[iE]
		if element=='Ge':
			iQ=int(np.floor((E-Emin)/3.0))
		if element=='Si':
			iQ=int(np.floor((E-Emin)/3.8))
		if iQ<n and E>Emin:
			for iq in range(0,len(q_vec)):
				q=q_vec[iq]
				vmin=E/q+q/(2*m_chi)
				if vmin<v_esc+v_E:
					eta0=etav0(vmin,v0,v_E,v_esc)*q/E
					integraltemp[0]=W1[iq,iE]*(DM_R.R1v0(q,E,cvec,j_x,u.electronmass,mu,m_chi)*eta0+DM_R.R1v2(q,E,cvec,j_x,u.electronmass,mu,m_chi)*etav2(vmin,v0,v_E,v_esc)*q/E)
					integraltemp[1]=ReW2[iq,iE]*DM_R.ReR2(q,E,cvec,j_x,u.electronmass,mu,m_chi)*eta0
					integraltemp[2]=ImW2[iq,iE]*DM_R.ImR2(q,E,cvec,j_x,u.electronmass,mu,m_chi)*eta0
					integraltemp[3]=W3[iq,iE]*DM_R.R3(q,E,cvec,j_x,u.electronmass,mu,m_chi)*eta0
					integraltemp[4]=W4[iq,iE]*DM_R.R4(q,E,cvec,j_x,u.electronmass,mu,m_chi)*eta0
					integraltemp[5]=W5[iq,iE]*DM_R.R5(q,E,cvec,j_x,u.electronmass,mu,m_chi)*eta0
					Qvec[iQ]+=np.sum(integraltemp)
	norm=rho*dq*dE*u.kg*u.yr/(m_chi*128*np.pi*m_chi**2*u.electronmass**2)/(2*(math.erf(v_esc/v0)-2*(v_esc/v0)*np.exp(-v_esc**2/v0**2)/np.sqrt(np.pi))*v_E)
	if element=='Ge':
		norm/=135.33E9
	if element=='Si':
		norm/=52.33E9
	Qvec*=norm
	return Qvec
