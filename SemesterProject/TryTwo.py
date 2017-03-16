import numpy as np
import theano
import lasagne
import theano.tensor as T
import pandas
from theano import *
from sklearn import preprocessing
import scipy
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

def load_dataset():
	return data.as_matrix(columns=["Month", "DayofMonth", "DayofWeek", "DepartureTime", "ArrivalTime", "Carrier", "FlightNo", "Origin", "Dest", "Distance"])
	
def load_expected_outputs():
	return data.as_matrix(columns=["ArrivalDelay"])

def compile():
    x, z = T.matrices('x', 'z')
    lh = lasagne.layers.InputLayer(shape=(None, x.shape[1]), input_var=x)
    ly = lasagne.layers.DenseLayer(lh, 6, nonlinearity=lasagne.nonlinearities.linear)
    y = lasagne.layers.get_output(ly)
    params = lasagne.layers.get_all_params(ly, trainable=True)
    cost = T.sum(lasagne.objectives.squared_error(y, z))
    updates = lasagne.updates.sgd(cost, params, learning_rate=0.0001)
    return theano.function([x, z], [y, cost], updates=updates)


def main():
    f_train = compile()

    x_train = load_dataset().astype(theano.config.floatX)
    y_train = load_expected_outputs()\
        .astype(theano.config.floatX)

    for _ in xrange(100):
        print f_train(x_train, y_train)


main()