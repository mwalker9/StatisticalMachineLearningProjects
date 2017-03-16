import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import seaborn

successes = 0

def p( x, t=1.0 ):
    return np.exp( -10*t*((x-2)**2) ) + 0.3*np.exp( -0.5*10*t*((x+1)**2) )

def getXPrime(x, scale):
	return np.random.normal(loc=x, scale=scale)

scales = np.asarray([.1, 1., 10])

scales=scales**.5

for scale in scales:
	timesteps = 10000
	successes = 0
	states = np.zeros((3, timesteps))

	for run in range(states.shape[0]):
		for t in range(timesteps - 1):
			x = states[run][t]
			xprime = getXPrime(x, scale)
			alpha = (p(xprime) * st.norm.pdf(x, xprime, 1.0)) / (p(x) * st.norm.pdf(xprime, x, 1.0))
			r = min(alpha, 1)
			u = np.random.uniform(0, 1.000001)
			assert u <= 1.0 and u >= 0
			if u<r:
				successes += 1
				states[run][t+1] = xprime
			else:
				states[run][t+1] = x

	plt.figure()
	plt.title("Trace for ss="+str(scale))
	for k in range(3):
		plt.plot(states[k, :])
	plt.ylabel("State")
	plt.xlabel("Time")
		
	actualDist = np.asarray([p(x) for x in np.arange(-3, 3, .1)])
	histArray = np.hstack((states[0, :], states[1, :], states[2, :]))

	print("Success Rate", successes/(timesteps*3.))

	plt.figure()
	plt.title("State Histogram for ss="+str(scale))
	plt.plot(np.arange(-3, 3, .1), actualDist)
	plt.hist(histArray, normed=True, bins=500)

plt.show()