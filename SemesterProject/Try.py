'''
Distance = 8.0029 * ElapsedTime - 289.9617
NeededTime = (Distance + 289.9617) / 8.0029
ExtraTime = AllottedTime - NeededTime 
'''
from theano.tensor.nnet import sigmoid
import pandas
from theano import *
from sklearn import preprocessing
import scipy
import numpy as np
from lasagne import layers
from lasagne.updates import nesterov_momentum
from lasagne.nonlinearities import softmax
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator
import lasagne
import nolearn
from lasagne.objectives import squared_error
from lasagne.updates import nesterov_momentum,adagrad

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer

data = pandas.read_csv("cleanedData.dat")

data.ArrivalTime = data.ArrivalTime.apply(lambda x: int(x.split(':')[0])*60 +int(x.split(':')[1]))

data.DepartureTime = data.DepartureTime.apply(lambda x: int(x.split(':')[0])*60 +int(x.split(':')[1]))

le_airline = preprocessing.LabelEncoder()

data.Carrier = le_airline.fit_transform(data.Carrier)

le_airport_origin = preprocessing.LabelEncoder()

data.Origin = le_airport_origin.fit_transform(data.Origin)

le_airport_dest = preprocessing.LabelEncoder()

data.Dest =  le_airport_dest.fit_transform(data.Dest)

data["ot"] = np.where(data.ArrivalDelay > 0, 0, 1)

def load_dataset():
	return data.as_matrix(columns=["Month", "DayofMonth", "DayofWeek", "DepartureTime", "ArrivalTime", "Carrier", "FlightNo", "Origin", "Dest", "Distance"])
	
def load_expected_outputs():
	return data.as_matrix(columns=["ot"])
	
def train():
	X = load_dataset() 
	y = load_expected_outputs() # assuming you have some data you're training on here, removed for brevity
	
	'''clf= NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden1', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None,X.shape[1]),
    hidden1_num_units=512,  # number of units in hidden layer
    output_num_units=1,  # target values

    # optimization method:
    update=adagrad,

    update_learning_rate = shared(np.float32(0.1)),

    batch_iterator_train=BatchIterator(batch_size=100),
    regression=True,  # flag to indicate we're dealing with regression problem
    max_epochs=100,  # we want to train this many epochs
    verbose=1,
    train_split=TrainSplit(eval_size=0.1)

    )'''
	i = InputLayer
	layers0 = [('input', i),
           ('dense0', DenseLayer),
           ('output', DenseLayer)]
	
	learning_rate = theano.shared(np.float32(.001))
	
	net0 = NeuralNet(layers=layers0, 
				input_shape=(None, X.shape[1]),
				 dense0_num_units=200,
				 dense0_b = None,
                 output_num_units=1,
				 output_nonlinearity=sigmoid,
                 regression=False,
                 objective_loss_function = squared_error,
                 update=nesterov_momentum,
                 update_learning_rate=learning_rate,
                 update_momentum=0.9,
				 train_split=nolearn.lasagne.TrainSplit(eval_size=0.1),
                 verbose=1,
                 max_epochs=3)
	
	
	#l1 = InputLayer(shape=(None, X.shape[1]))
	#l2 = DenseLayer(l1, num_units=y.shape[1], nonlinearity=None)
	#net = NeuralNet([("inputLayer",l1), ("output", l2)], regression=True, update_learning_rate=0.01, input_shape=(None, y.shape[1]))
	net0.fit(X, y)
	return net0
	
def run():
	load_dataset()