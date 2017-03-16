#hmc
from autograd import grad
import autograd.numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

tries = 0.
successes = 0.

def p( x, t=1.0 ):
    return np.exp( -10*t*((x-2)**2) ) + 0.3*np.exp( -0.5*10*t*((x+1)**2) )
	
def U(q):
	return -np.log(p(q))	

def HMC(U, grad_U, epsilon, L, current_q):
	q = current_q
	p = np.random.normal(loc=0, scale=2)
	current_p = p
	p = p - epsilon * grad_U(q) / 2.
	
	for i in range(L):
		q = q + epsilon * p
		if i != L - 1:
			p = p - epsilon * grad_U(q)
	
	p = p - epsilon * grad_U(q) / 2.
	p = -p
	
	current_U = U(current_q)
	current_K = (current_p ** 2) / 2.
	proposed_U = U(q)
	proposed_K = (p**2) / 2
	
	runif = np.random.uniform(0, 1.000001)
	assert runif <= 1.0 and runif >= 0
	if runif < np.exp(current_U - proposed_U + current_K - proposed_K):
		return q
	else:
		return current_q
		

timesteps = 10000

states = np.zeros((3, timesteps))

for run in range(states.shape[0]):
	for t in range(timesteps - 1):
		tries += 1
		states[run][t + 1] = HMC(U, grad(U), .5, 1, states[run][t]) 
		if states[run][t+1] != states[run][t]:
			successes += 1
		
plt.figure()
plt.title("Trace for momentum scale = 2.0")
for k in range(3):
	plt.plot(states[k, :])
plt.ylabel("State")
plt.xlabel("Time")
	
actualDist = np.asarray([p(x) for x in np.arange(-3, 3, .1)])
histArray = np.hstack((states[0, :], states[1, :], states[2, :]))

print("Success Rate", successes/tries)

plt.figure()
plt.title("State Histogram")
plt.plot(np.arange(-3, 3, .1), actualDist)
plt.hist(histArray, normed=True, bins=500)
plt.show()