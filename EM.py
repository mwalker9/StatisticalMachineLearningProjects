import matplotlib.pyplot as plt
import numpy as np
import scipy.io as IO
from scipy.stats import multivariate_normal
import seaborn
%matplotlib inline
def gaussian(x, mean, sigma):
	return multivariate_normal.pdf(x, mean=mean, cov=sigma)
	
def getData():
	mat = IO.loadmat('old_faithful.mat')
	return mat
	
def cov_to_pts(cov):
    circ = np.linspace( 0, 2*np.pi, 100 )
    sf = np.asarray( [ np.cos( circ ), np.sin( circ ) ] )
    [u,s,v] = np.linalg.svd( cov )
    pmat = u*2.447*np.sqrt(s) # 95% confidence
    return np.dot(  pmat, sf )

def EM(mus, covs, mws, data):
	iteration = 0
	originalMus = []
	originalCovs = [] 
	originalMws = []
	while not np.array_equal(originalMus, mus) or not np.array_equal(originalCovs, covs):
		originalMus = mus
		originalCovs = covs
		originalMws = mws
		#compute Responsiblities w_ik
		w = np.zeros((len(data),2), np.float64)
		for i in range(len(data)):
			for k in range(2): #Number of clusters
				w[i][k] = gaussian(data[i], mus[k], covs[k])*mws[k] 
			# Then normalize over all clusters
			w[i] = w[i]/np.sum(w[i])
		#update mixing weights mws
		N = [0., 0.]
		for i in range(len(data)):
			for k in range(2):
				N[k] = N[k] + w[i][k]
		mws[0] = float(N[0])/len(data)
		mws[1] = float(N[1])/len(data)
		for k in range(2):
			sum = np.asarray([0., 0.])
			for i in range(len(data)):
				sum = sum + (1./N[k]) * (w[i][k]  * data[i])
			mus[k] = sum
		mus = np.asarray(mus)
		#update covariance
		covs = list()
		for k in range(2):
			sum = np.asarray([[0.,0.], [0.,0.]])
			for i in range(len(data)):
				temp = (data[i] - mus[k])
				temp = np.atleast_2d(temp)
				sum = sum + w[i][k] * temp.T * temp
			sum = sum/N[k]
			covs.append(sum)
		if (iteration) % 4 == 0:
			plt.figure(iteration / 4 + 1)
			plt.title("After iteration "+str(iteration+1))
			plt.xlim(-3, 3)
			plt.ylim(-40, 30)
			c = cov_to_pts(covs[0])
			c[0] = c[0] + mus[0][0]
			c[1] = c[1] + mus[0][1]
			plt.plot(c[0], c[1], label="Class One 95th Percentile")
			c = cov_to_pts(covs[1])
			c[0] = c[0] + mus[1][0]
			c[1] = c[1] + mus[1][1]
			plt.plot(c[0], c[1], label="Class Two 95th Percentile")
			colors = [str(item) for item in w[:, 0]]
			#print(colors)
			plt.scatter(data[:, 0], data[:, 1], color=colors, s=100)
			plt.scatter(mus[0,0], mus[0,1], color = 'w', s=500)
			plt.scatter(mus[1,0], mus[1,1], color = 'k', s=500)
			plt.legend(loc=3)
			plt.gray()
			plt.show()
		iteration = iteration + 1

mus = np.asarray( [[-1.17288986, -0.11642103],
				   [-0.16526981,  0.70142713]])
# the Gaussian covariance matrices
covs = list()
covs.append( 
	np.asarray([[ 0.74072815,  0.09252716],
				[ 0.09252716,  0.5966275 ]]))
covs.append( 
	np.asarray([[ 0.39312776, -0.46488887],
				[-0.46488887,  1.64990767]]) )
 
# The Gaussian mixing weights
mws = [ 0.68618439, 0.31381561 ]
data = np.asarray(getData()["data"])
data[:, 0] = data[:, 0] - data[:, 0].mean()
data[:, 1] = data[:, 1] - data[:, 1].mean()
EM(mus, covs, mws, data)