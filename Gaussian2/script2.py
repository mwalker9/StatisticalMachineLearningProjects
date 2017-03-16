import numpy
import pandas
from sklearn.cluster import MiniBatchKMeans
import random
import math
import matplotlib.pyplot as plt
%matplotlib inline

storeData = pandas.read_csv('store.csv')

def StoreTypeK(x1, x2):
	if x1 == x2:
		return 1
	distance = .6
	return numpy.exp(-distance)

def AssortmentK(x1, x2):
	distance = 1
	if x1 == x2:
		distance = 1
	elif x1 != 'a' and x2 != 'a':
		distance = .7
	else:
		distance = .4
	return (1 + distance) ** 2

def CompetitionDist(x1, x2):
	if math.isnan(x1) or math.isnan(x2):
		return 0
	distance = abs(x1 - x2)
	return numpy.exp(-distance)
	
def Promo2(x1, x2):
	if x1 == x2:
		return 1
	else:
		return 0

def storeKernel(x1In, x2In):
	x1 = storeData.iloc[x1In-1]
	x2 = storeData.iloc[x2In-1]
	return 1.2*StoreTypeK(x1.StoreType, x2.StoreType) + 1.2*AssortmentK(x1.Assortment, x2.Assortment) + .8*CompetitionDist(x1.CompetitionDistance, x2.CompetitionDistance) + 1.4*Promo2(x1.Promo2, x2.Promo2)

def mod(x, m):
	if m<0:
		m = -m
	r = x % m
	if r<0:
		return r+m
	else:
		return r
	
def modularSubtraction(a, b, m=7):
	return min(mod(a-b, m), mod(b-a, m))
		
def dowKernel(x1, x2):
	x1 = x1 - 1
	x2 = x2 - 1
	distance = modularSubtraction(x1, x2)
	return numpy.exp(-distance)

def dateKernel(x1, x2):
	doy1 = x1.day 
	doy2 = x2.day 
	distance = abs(doy1-doy2)
	mon1 = x1.month - 1
	mon2 = x2.month - 1
	monthDistance = modularSubtraction(mon1, mon2, m=12)
	return .2 * numpy.exp(-distance) + .8 * numpy.exp(-monthDistance)

def promoKernel(x1, x2):
	if x1 == x2:
		return 1
	else:
		return 0

def stKernel(x1, x2):
	if x1 == x2:
		return 1
	elif x1 == 'a' and x2 == 'b':
		return .75
	elif x1 == 'b' and x2 == 'a':
		return .75
	elif x1 =='c' and x2 in ['a','b']:
		return .6
	elif x1 in ['a', 'b'] and x2 == 'c':
		return .6
	else:
		return .3

def scKernel(x1, x2):
	if x1 == x2:
		return 1
	else:
		return 0

def kernel(x1, x2):
	return .75 * storeKernel(x1.Store, x2.Store) + dowKernel(x1.DayOfWeek, x2.DayOfWeek) + .75 * dateKernel(x1.Date, x2.Date) + promoKernel(x1.Promo, x2.Promo) + 3 * stKernel(x1.StateHoliday, x2.StateHoliday) + 2*scKernel(x1.SchoolHoliday, x2.SchoolHoliday)
	
def K(x1, x2, pr=False):
	dim1 = 1
	dim2 = 1
	if len(x1.shape) == 2:
		dim1 = x1.shape[0]
	if len(x2.shape) == 2:
		dim2 = x2.shape[0]
	mat = numpy.zeros((dim1, dim2))
	ctr = 0
	total = dim1 * dim2
	oldPercentage = 0
	for i in range(dim1):
		for j in range(dim2):
			if dim1 != 1 and dim2 != 1:
				mat[i][j] = kernel(x1.iloc[i], x2.iloc[j])
			elif dim1 == 1 and dim2 == 1:
				mat[i][j] = kernel(x1, x2)
			elif dim1 == 1 and dim2 != 1:
				mat[i][j] = kernel(x1, x2.iloc[j])
			else:
				mat[i][j] = kernel(x1.iloc[i], x2)
			newPercentage = round(( (1.0*ctr)/(1.0*total)), 2)
			if oldPercentage != newPercentage and pr:
				print "calculating matrix %.2f" % newPercentage
				oldPercentage = newPercentage
			ctr = ctr+1
	return mat
 
def getMRegressors(data, m):
	tempData = data.replace(to_replace='a', value=1)
	tempData = tempData.replace(to_replace='b', value=2)
	tempData = tempData.replace(to_replace='c', value=3)
	del tempData["Date"]
	kmeans = MiniBatchKMeans(n_clusters=m/2, batch_size=1000)
	labels = kmeans.fit_predict(tempData)
	idxs = []
	for i in range(m/2):
		for k in numpy.random.permutation(len(labels)):
			if labels[k] == i:
				idxs.append(k)
				if len(idxs) % 2 == 0:
					break
	return idxs
	
def getXmn(data):
	m = 100 #THIS IS THE M YOU'RE LOOKING FOR
	#n = data.shape[0]
	data = data[data['Sales']!=0]
	idxs = getMRegressors(data, m)
	#otherIdxs = random.sample(xrange(data.shape[0]), n)
	data["Date"] = pandas.to_datetime(data["Date"])
	return data.iloc[idxs], data

def predictMean(cached, pval, Xm):
	return numpy.dot(K(pval, Xm), cached)
	
def run():
	# this is our training data
	data = pandas.read_csv( 'store_train.csv' )
	# these are what we need to make predictions for
	testd = pandas.read_csv( 'store_test.csv' )
	 
	N = testd.shape[0]
	testd["Date"] = pandas.to_datetime(testd["Date"])
	my_preds = numpy.zeros(( N, 1 ))
	print('Getting all regressors')
	Xm, Xn = getXmn(data)
	Y = Xn.Sales
	sigma = 1
	print('Figuring out covariance between m and n regressors')
	Kmn = K(Xm, Xn, pr=True)
	Knm = Kmn.T
	print('precalculating matrix')
	cached = numpy.linalg.pinv(numpy.dot(Kmn, Knm) + sigma**2 * K(Xm, Xm))
	print('precalculating matrix part 2')
	cached = numpy.dot(cached, numpy.dot(Kmn, Y))
	print('starting predictions')
	for id in range( 0, N ):
		
		# grab a data element one at a time
		pval = testd.iloc[id]
		sid = pval.Store
		dow = pval.DayOfWeek
	 
		if pval.Open == 0:
			#most stores are closed on Sunday.  Awesome!
			my_preds[ id, 0 ] = 0
		else:
			my_preds[ id, 0 ] = predictMean(cached, pval, Xm)
	 
		# a little "progress bar"
		print "%.2f (%d/%d)" % ( (1.0*id)/(1.0*N), id, N )
	 
	 
	sfile = open( 'mean_sub.csv', 'wb' )
	sfile.write( '"Id","Sales"\n' )
	for id in range( 0, N ):
		sfile.write( '%d,%.2f\n' % ( id+1, my_preds[id] ) )
	sfile.close()
	
data = pandas.read_csv( 'store_train.csv' )
Xm, Xn = getXmn(data)
Kmm = K(Xm, Xm)
fig = plt.figure("Covariance Matrix")
ax = fig.add_subplot(111)
cax = ax.matshow(Kmm, interpolation='nearest')
fig.colorbar(cax)
plt.show()
print("I submitted the results of this to Kaggle and got a MSE of 0.40097 which would have only been good enough for #3049")
