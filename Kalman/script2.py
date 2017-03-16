import numpy as np
import scipy.io
import skimage.feature
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

def cov_to_pts(cov):
    circ = np.linspace( 0, 2*np.pi, 100 )
    sf = np.asarray( [ np.cos( circ ), np.sin( circ ) ] )
    [u,s,v] = np.linalg.svd( cov )
    pmat = u*2.447*np.sqrt(s) # 95% confidence
    return np.dot(  pmat, sf )

tmp = scipy.io.loadmat('ball_data.mat')
frames = tmp['frames']  # frames of our video
ball = tmp['ball']  # our little template
 
data = []
for i in range( 0, frames.shape[1] ):
    tmp = np.reshape( frames[:,i], (360,243) ).T  # slurp out a frame
    ncc = skimage.feature.match_template( tmp, ball )  # create a normalized cross correlation image
    maxloc = np.unravel_index( tmp.argmax(), tmp.shape )  # find the point of highest correlation
    data.append( maxloc )  # record the results
 
data = np.asarray( data )

'''
Our position is equal to the old position plus our velocity plus 1/2 the acceleration
The velocity is equal to the old velocity plus our acceleration
The acceleration is constant-ish -- it's just decreasing due to friction, which we'll account for in our
dynamics uncertainty
'''
A = np.asarray([
    [ 1, 0, 1, 0, 0.5, 0 ],
    [ 0, 1, 0, 1, 0, 0.5 ],
    [ 0, 0, 1, 0, 1, 0 ],
    [ 0, 0, 0, 1, 0, 1 ],
    [ 0, 0, 0, 0, 1, 0 ],
    [ 0, 0, 0, 0, 0, 1 ] ])
	
# our observations are only the position components -- so this keeps it that way
C = np.asarray([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0]])
 
# our dynamics noise tries to force the acceleration to account for more of the
# dynamic uncertainty -- most of the uncertainty comes from friction
# Also, uncertainty comes from camera movement--which I consider uncertainty in velocity
# I don't believe the dynamics are that uncertain, so I'm scaling this down a lot
Q = 1e-5 * np.eye( 6 )
Q[2,2] = Q[2,2] * 5
Q[3,3] = Q[3,3] * 5
Q[4,4] = Q[4,4] * 10
Q[5,5] = Q[5,5] * 10
 
'''
Since the data is very bad, especially near the end, we'll have this
be scaled by a rather large factor
'''
R = 2000 * np.eye( 2 )
'''
Furthermore, the y-values seem to be off more than the x values
'''
R[1, 1] = R[1,1] * 3

# initial state
mu_t = np.zeros(( 6, 1 ))
#The average x-value and y-value we will assume is right at the middle of the picture
mu_t[0, 0] = 175
mu_t[1, 0] = 125
'''
The initial point is in the middle plus or minus about 150, so that's why
I multiplied this by 150
'''
sigma_t = np.eye( 6 ) * 150
mus = list()
mus.append(mu_t)
covs = list()
for i in range(len(data)):
	sigmaprime_tminus1 = np.dot(np.dot(A, sigma_t), A.T) + Q
	S_t = np.dot(np.dot(C, sigmaprime_tminus1), C.T) + R
	K_t = np.dot(np.dot(sigmaprime_tminus1, C.T), np.linalg.pinv(S_t))
	sigma_t = sigmaprime_tminus1 - np.dot(np.dot(K_t, C), sigmaprime_tminus1)
	covs.append(sigma_t)
	yhat_t = np.dot(C, np.dot(A, mu_t))
	mu_t = np.dot(A, mu_t) + np.dot(K_t, np.atleast_2d(data[i]).T - yhat_t)
	mus.append(mu_t)

mus = np.asarray(mus)
plt.figure()
mus = mus[:,:,0]
plt.plot(mus[:, 0], mus[:, 1], label="Predicted locations")
#plt.gca().invert_yaxis()
plt.legend()
plt.show()
    
for t in range(0, data.shape[0]):
    tmp = np.reshape( frames[:,t], (360,243) ).T
 
    plt.figure(1)
    plt.clf()
    plt.imshow(tmp, interpolation='nearest', cmap=matplotlib.cm.gray )
    plt.scatter( data[t][1], data[t][0] )
    plt.scatter( mus[t][1], mus[t][0] )
 
    foo = cov_to_pts( covs[t][0:2,0:2] )
	
    plt.plot( foo[0,:] + mus[t][1], foo[1,:] + mus[t][0] )
    plt.xlim([1, 360])
    plt.ylim([243,1])
    if t in [53, 55, 77, 105, 106]: #For zero indexing
		plt.show()