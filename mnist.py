import sklearn.metrics
import scipy.io
import numpy
import matplotlib.pyplot as plt
import matplotlib
from scipy import spatial
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def gaus(x, x0):
	diff = x-x0
	return numpy.exp(-numpy.dot(diff, diff))

def getProbabilityInDistribution(trainingData, novelData):
	gaussians = []
	for imgIdx in range(trainingData.shape[1]):
		gaussians.append(gaus(trainingData[:, imgIdx], novelData))
	return sum(gaussians)/trainingData.shape[0]
		

def getNumbers(data, labels, number):
	idxs = []
	for i in range(len(labels == number)):
		if (labels == number)[i][0]:
			idxs.append(i)
	return data[:, idxs]
	
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    #tick_marks = numpy.arange(len(iris.target_names))
    #plt.xticks(tick_marks, iris.target_names, rotation=45)
    #plt.yticks(tick_marks, iris.target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
	

train_mat = scipy.io.loadmat('mnist_train.mat')
train_data = train_mat['images']
train_labels = train_mat['labels']

zeroes = getNumbers(train_data, train_labels, 0)
ones = getNumbers(train_data, train_labels, 1)
twos = getNumbers(train_data, train_labels, 2)
threes = getNumbers(train_data, train_labels, 3)
fours = getNumbers(train_data, train_labels, 4)
fives = getNumbers(train_data, train_labels, 5)
sixes = getNumbers(train_data, train_labels, 6)
sevens = getNumbers(train_data, train_labels, 7)
eights = getNumbers(train_data, train_labels, 8)
nines = getNumbers(train_data, train_labels, 9)
	
classes = []
classes.append(zeroes)
classes.append(ones)
classes.append(twos)
classes.append(threes)
classes.append(fours)
classes.append(fives)
classes.append(sixes)
classes.append(sevens)
classes.append(eights)
classes.append(nines)
classifications = []
	
averageImages = []
for currentClass in classes:
	averageImages.append(numpy.mean(currentClass, axis = 1))
dist = spatial.KDTree(averageImages)

i = 0
for k in averageImages:
	plt.figure(i)
	i = i + 1
	plt.imshow(k.reshape(28,28).T, interpolation='nearest', cmap=matplotlib.cm.gray)

test_mat = scipy.io.loadmat('mnist_test.mat')
test_data = test_mat['t10k_images']
test_labels = test_mat['t10k_labels']

classifications = []
for k in range(test_data.shape[1]):
	classifications.append(dist.query(test_data[:, k])[1])#We're just finding the closest "prototype" and using that because in the end the probability
	#is very dependent on the variance we use--but we're just maximizing the probability and since the probability increases with a decrease in distance
	#we can just find the least distance--which is what I've done here
print("Just using the means we have an error rate of: ")
print(str((1. - accuracy_score(test_labels, classifications))*100) + "%")

cm = confusion_matrix(test_labels, classifications)
plot_confusion_matrix(cm)
	
classifications = []
numberOfInstances = 1000
for k in range(numberOfInstances):
	probabilities = []
	for currentclass in classes:
		probabilities.append(getProbabilityInDistribution(currentclass, test_data[:, k]))
	classifications.append(numpy.argmax(probabilities))
	
print("Using KDE we have an error rate of: ")
print(str((1. - accuracy_score(test_labels[:numberOfInstances], classifications))*100) + "%")
cm = confusion_matrix(test_labels[:numberOfInstances], classifications)
plot_confusion_matrix(cm)