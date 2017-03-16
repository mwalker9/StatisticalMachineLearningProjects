import lab8_common as common
import scipy.io as SIO
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.stats
%matplotlib inline
def ray_lik(sensorData, particleData): #returns Nx1 Matrix with likelihood of each particle
	difference = np.sum(abs(sensorData - particleData), axis=1)
	result = scipy.stats.norm(scale=.15).pdf(difference)
	return result

def proposalDist(parts):
	twiddler = np.random.rand(parts.shape[0], parts.shape[1])
	twiddler = twiddler - (np.ones((3, twiddler.shape[1])) * .5)
	twiddler = twiddler * 2 #Now we have [-1, 1]
	twiddler = twiddler * .05
	twiddler[2, :] = twiddler[2, :] * 15
	result = parts + twiddler
	top = result[0:2, :]
	top[(top > 1)] = 1
	top[(top < 0)] = 0
	result[2, :] = result[2, :] % (2 * np.pi)
	result = np.vstack((top, result[2, :]))
	return result
	
	
def sampleOverDistribution(weighting, particles):
	results = np.random.mtrand.multinomial(particles.shape[1], weighting)
	new_particles = np.zeros((3, particles.shape[1]))
	currentParticle = 0
	currentCounter = 0
	for i in range(particles.shape[1]):
		while currentCounter >= results[currentParticle]:
			currentParticle = currentParticle + 1
			currentCounter = 0
		new_particles[:, i] = particles[:, currentParticle]
		currentCounter = currentCounter + 1
	return new_particles

N = 10000
data = SIO.loadmat("sensors.mat")
true_states = data["true_states"]
sensors = data["sonars"]

room_map = common.create_map()

for i in range( 0, room_map.shape[0] ):
    plt.plot( [room_map[i,0], room_map[i,2]], [room_map[i,1], room_map[i,3]] )

parts = np.zeros((3, N))
#Initialize the particles
for i in range(N):
	parts[0, i] = random.random()
	parts[1, i] = random.random()
	parts[2, i] = random.random() * np.pi * 2

exs = np.zeros((3, sensors.shape[1]))
	
for t in range(sensors.shape[1]):
	parts = proposalDist(parts)
	part_sensor_data = common.cast_rays( parts, room_map )
	likelihoods = ray_lik(sensors[:,t:t+1].T, part_sensor_data).T
	likelihoods = likelihoods/sum(likelihoods)
	exs[:, t] = np.sum(likelihoods * parts, axis=1)
	parts = sampleOverDistribution(likelihoods, parts)

plt.plot(true_states[0, :], true_states[1, :], color='b', label="True States")
plt.plot(exs[0, :], exs[1, :], color='r', label="Expected Values")
plt.scatter(parts[0,:], parts[1,:])
plt.title('Room Map')
plt.legend()
plt.show()