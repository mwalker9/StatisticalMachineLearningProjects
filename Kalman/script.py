import scipy.io as SIO
import numpy as np
import matplotlib.pyplot as plt
import seaborn
%matplotlib inline

def cov_to_pts(cov):
    circ = np.linspace( 0, 2*np.pi, 100 )
    sf = np.asarray( [ np.cos( circ ), np.sin( circ ) ] )
    [u,s,v] = np.linalg.svd( cov )
    pmat = u*2.447*np.sqrt(s) # 95% confidence
    return np.dot(  pmat, sf )

data = SIO.loadmat('kfdata.mat')
true_data = data['true_data']
data = data['data']
A = np.asarray([
    [ 1, 0, 1, 0, 0.5, 0 ],
    [ 0, 1, 0, 1, 0, 0.5 ],
    [ 0, 0, 1, 0, 1, 0 ],
    [ 0, 0, 0, 1, 0, 1 ],
    [ 0, 0, 0, 0, 1, 0 ],
    [ 0, 0, 0, 0, 0, 1 ] ])
 
# our observations are only the position components
C = np.asarray([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0]])
 
# our dynamics noise tries to force random accelerations to account
# for most of the dynamics uncertainty
Q = 1e-2 * np.eye( 6 )
Q[4,4] = 0.5  # variance of accelerations is higher
Q[5,5] = 0.5
 
# our observation noise
R = 20 * np.eye( 2 )

# initial state
mu_t = np.zeros(( 6, 1 ))
sigma_t = np.eye( 6 )
mus = list()
mus.append(mu_t)
for i in range(len(data)):
	sigmaprime_tminus1 = np.dot(np.dot(A, sigma_t), A.T) + Q
	S_t = np.dot(np.dot(C, sigmaprime_tminus1), C.T) + R
	K_t = np.dot(np.dot(sigmaprime_tminus1, C.T), np.linalg.pinv(S_t))
	sigma_t = sigmaprime_tminus1 - np.dot(np.dot(K_t, C), sigmaprime_tminus1)
	yhat_t = np.dot(C, np.dot(A, mu_t))
	mu_t = np.dot(A, mu_t) + np.dot(K_t, np.atleast_2d(data[i]).T - yhat_t)
	mus.append(mu_t)
	pts = cov_to_pts(sigma_t[:2, :2])
	pts[0] = pts[0] + mu_t[0][0]
	pts[1] = pts[1] + mu_t[1][0]
	plt.plot(pts[0], pts[1])
mus = np.asarray(mus)
plt.plot(true_data[:,0], true_data[:,1], label="True Position")
plt.plot(mus[:, 0, :], mus[:, 1, :], label="Predicted locations")
plt.scatter(data[:, 0], data[:, 1], label="Observed data")
plt.legend()
plt.show()