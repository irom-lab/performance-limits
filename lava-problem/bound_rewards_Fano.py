import numpy as np

# Definition of Fano bound
def bound_Fano(R_perp, I):
    return -(I+np.log(2-R_perp))/np.log(R_perp)

def compute_Fano(nx, nu, ny, T, p0, px_x, py_x, R, R0_expected):
	'''
    One step Fano bound (T=1)
    '''
	# Compute R_perp
	R_u = p0[0] * R
	Rt_perp = np.max(R_u)

    # Compute mutual information
	px_0 = p0 # Probability of x_0
	pyx_0 = py_x*px_0[None,:]  # pyx(i,j) is probability of measurement i and state j 
	py_0 = np.sum(pyx_0, 1)

	I = 0.0
	for ii in range(0,ny):
		for jj in range(0,nx):
			if (np.abs(pyx_0[ii,jj]) > 1e-5):
				# mutual information using KL divergence
				I = I + pyx_0[ii,jj]*np.log(pyx_0[ii,jj]/(py_0[ii]*px_0[jj]))
	bound = bound_Fano(Rt_perp, I)
	
	return bound

