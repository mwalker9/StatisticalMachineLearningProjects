import numpy
import seaborn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
%matplotlib inline
def delta(x, y):
	if x == y:
		return 1
	else:
		return 0
		
def addNoise(x, y, normalKernelFunction, idx1, idx2):
	sigmaSquared = .1
	return normalKernelFunction(x, y, idx1, idx2) + sigmaSquared * delta(idx1, idx2)

def linearKernel(x1, x2, idx1, idx2):
	return x1 * x2
	
def linearWithNoise(x1, x2, idx1, idx2):
	return addNoise(x1, x2, linearKernel, idx1, idx2)
	
def RBFKernel(x1, x2, idx1, idx2):
	sigma = 1#The predicted line becomes more spikey with a smaller sigma, and more smooth with a larger sigma
	dist = x1 - x2
	distSquared = dist ** 2
	twoSigmaSquared = 2 * (sigma ** 2)
	return numpy.exp(-distSquared/twoSigmaSquared)

def RBFWithNoise(x1, x2, idx1, idx2):
	return addNoise(x1, x2, RBFKernel, idx1, idx2)

def polynomialKernel(x1, x2, idx1, idx2):
	gamma = 1
	r = 1
	m = 3 #The variance increases very much with a higher m -- in the limit, it becomes basically a line when m is smaller
	return (r + gamma * x1 * x2) ** m

def polynomialWithNoise(x1, x2, idx1, idx2):
	return addNoise(x1, x2, polynomialKernel, idx1, idx2)
	
def getUpperLeft(xVals, kernelFunction):
	matrix = numpy.zeros((len(xVals), len(xVals)), numpy.float64)
	for i in range(len(xVals)):
		for k in range(len(xVals)):
			matrix[i][k] = kernelFunction(xVals[i], xVals[k], i, k)
	return matrix
	
def getRow(xVals, xPrime, kernelFunction):
	row = numpy.zeros(len(xVals), numpy.float64)
	for i in range(len(xVals)):
		row[i] = kernelFunction(xVals[i], xPrime, 1, 0)
	return row
		
	
def dataSet():
	data_xvals = numpy.asarray( [ 1.0, 3.0, 5.0, 6.0, 7.0, 8.0 ] )
	data_yvals = numpy.sin( data_xvals )
	return data_xvals, data_yvals
	
def go(kernelFunction, x, y, pltTitle, figNumber):
	covariance = getUpperLeft(x, kernelFunction)
	mean = []
	var = []
	domain = numpy.arange(-2, 10, .1)
	for xprime in domain:
		row = getRow(x, xprime, kernelFunction)
		inverse = numpy.linalg.pinv(covariance)
		mean.append(numpy.dot(numpy.dot(row, inverse), y))
		var.append(kernelFunction(xprime, xprime, 1, 1) - numpy.dot(numpy.dot(row, inverse), numpy.transpose(row)))
	mean = numpy.asarray(mean)
	var = numpy.asarray(var)
	y1 = mean - var
	y2 = mean + var
	plt.figure(figNumber).suptitle(pltTitle, fontsize=12)
	plt.plot(domain, mean, color="black", label="Mean")
	plt.scatter(x, y, color="black")
	plt.gca().fill_between(domain, y1, y2, where=y2 >= y1, facecolor='blue', interpolate=True, alpha=.25, label="Variance")
	plt.legend()
	
def polynomialRun(x, y):
	go(polynomialKernel, x, y, 'No Noise Gaussian Process Regression With Polynomial Kernel', 1)
	go(polynomialWithNoise, x, y, 'Noisy Gaussian Process Regression With Polynomial Kernel', 2)

def RBFKernelRun(x, y):
	go(RBFKernel, x, y, 'No Noise Gaussian Process Regression With RBFKernel', 3)
	go(RBFWithNoise, x, y, 'Noisy Gaussian Process Regression With RBFKernel', 4)

def linearKernelRun(x, y):
	go(linearKernel, x, y, 'No Noise Gaussian Process Regression With Linear Kernel', 5)
	go(linearWithNoise, x, y, 'Noisy Gaussian Process Regression With Linear Kernel', 6)


xTrain, yTrain = dataSet()
polynomialRun(xTrain, yTrain)
RBFKernelRun(xTrain, yTrain)
linearKernelRun(xTrain, yTrain)