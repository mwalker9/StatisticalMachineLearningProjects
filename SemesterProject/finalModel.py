from sklearn import preprocessing
import pandas
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
	
def trainAndTest():
	data = pandas.read_csv("cleanedData.dat")

	data = data.sample(500000)
	
	data.ArrivalTime = data.ArrivalTime.apply(lambda x: int(x.split(':')[0])*60 +int(x.split(':')[1]))

	data.DepartureTime = data.DepartureTime.apply(lambda x: int(x.split(':')[0])*60 +int(x.split(':')[1]))

	le_airline = preprocessing.LabelEncoder()

	data.Carrier = le_airline.fit_transform(data.Carrier)

	le_airport_origin = preprocessing.LabelEncoder()

	data.Origin = le_airport_origin.fit_transform(data.Origin)

	le_airport_dest = preprocessing.LabelEncoder()

	data.Dest =  le_airport_dest.fit_transform(data.Dest)

	data["ot"] = np.where(data.ArrivalDelay > 0, 0, 1)

	data, test = train_test_split(data, test_size = 0.2)

	onTime = data["ot"].as_matrix()
	longFlights = data.where(data["Distance"]>2000).dropna()
	shortFlights = data.where(data["Distance"]<=2000).dropna()
	longFlightResponses = longFlights.ArrivalDelay.as_matrix()
	shortFlightResponses = shortFlights.ArrivalDelay.as_matrix()
	del longFlights["ArrivalDelay"]
	del shortFlights["ArrivalDelay"]
	shortOT = shortFlights["ot"]
	longOT = longFlights["ot"]
	longFlights = longFlights.as_matrix(columns=["Month", "DayofMonth", "DayofWeek", "DepartureTime", "ArrivalTime", "Carrier", "FlightNo", "Origin", "Dest", "Distance"])
	shortFlights = shortFlights.as_matrix(columns=["Month", "DayofMonth", "DayofWeek", "DepartureTime", "ArrivalTime", "Carrier", "FlightNo", "Origin", "Dest", "Distance"])
	#print(shortFlights.shape)
	#print(longFlights.shape)
	
	shortFlightsd = RandomForestClassifier(n_jobs=-1)
	shortFlightsd.fit(shortFlights, shortOT)
	
	print("First forest")
	
	longFlightsd = RandomForestClassifier()
	longFlightsd.fit(longFlights, longOT)
	
	print("Second forest")
	
	onTimeFlights = data.where(data["ot"] == 1).dropna()
	
	lateFlights = data.where(data["ot"] == 0).dropna()
	
	onTimeAnswer = onTimeFlights.ArrivalDelay.as_matrix()
	
	lateAnswer = lateFlights.ArrivalDelay.as_matrix()
	
	lateFlights = lateFlights.as_matrix(columns=["Month", "DayofMonth", "DayofWeek", "DepartureTime", "ArrivalTime", "Carrier", "FlightNo", "Origin", "Dest", "Distance"])
	onTimeFlights = onTimeFlights.as_matrix(columns=["Month", "DayofMonth", "DayofWeek", "DepartureTime", "ArrivalTime", "Carrier", "FlightNo", "Origin", "Dest", "Distance"])
	
	
	onTimeRegressor = LinearRegression()
	onTimeRegressor.fit(onTimeFlights, onTimeAnswer)
	
	print("First regressor")
	
	lateRegressor = LinearRegression()
	lateRegressor.fit(lateFlights, lateAnswer)
	
	print("Second regressor")
	#print(test.shape)

	longFlights = test.where(test["Distance"]>2000).dropna()
	
	answersLongFlights = longFlights["ArrivalDelay"].as_matrix()
	
	shortFlights = test.where(test["Distance"]<=2000).dropna()
	
	answersShortFlights = shortFlights["ArrivalDelay"].as_matrix()
	

	shortFlightsData = shortFlights.as_matrix(columns=["Month", "DayofMonth", "DayofWeek", "DepartureTime", "ArrivalTime", "Carrier", "FlightNo", "Origin", "Dest", "Distance"])
	
	longFlightsData = longFlights.as_matrix(columns=["Month", "DayofMonth", "DayofWeek", "DepartureTime", "ArrivalTime", "Carrier", "FlightNo", "Origin", "Dest", "Distance"])
	
	longIndicators = longFlightsd.predict(longFlightsData)
	
	shortIndicators = shortFlightsd.predict(shortFlightsData)
	
	shortOnTimeIdxs = np.where(shortIndicators == 1)[0]
	
	longOnTimeIdxs = np.where(longIndicators == 1)[0]
	
	shortOnTimePredictions = onTimeRegressor.predict(shortFlightsData[shortOnTimeIdxs, :])
	shortOnTimeAnswers = answersShortFlights[shortOnTimeIdxs]
	
	longOnTimePredictions = onTimeRegressor.predict(longFlightsData[longOnTimeIdxs, :])
	longOnTimeAnswers = answersLongFlights[longOnTimeIdxs]
	
	shortLateIdxs = np.where(shortIndicators == 0)[0]
	shortLateAnswers = answersShortFlights[shortLateIdxs]
	
	longLateIdxs = np.where(longIndicators == 0)[0]
	longLateAnswers = answersLongFlights[longLateIdxs]
	
	shortLatePredictions = lateRegressor.predict(shortFlightsData[shortLateIdxs, :])
	
	longLatePredictions = lateRegressor.predict(longFlightsData[longLateIdxs, :])
	
	predictions = np.hstack((shortOnTimePredictions, longOnTimePredictions, shortLatePredictions, longLatePredictions))
	
	answers = np.hstack((shortOnTimeAnswers, longOnTimeAnswers, shortLateAnswers, longLateAnswers))
	
	print(np.sqrt(mean_squared_error(predictions, answers)))
	
	shortOnTimeOT = shortFlights.ot.iloc[shortOnTimeIdxs]
	longOnTimeOT = longFlights.ot.iloc[longOnTimeIdxs]
	
	longLateOT = longFlights.ot.iloc[longLateIdxs]
	shortLateOT = shortFlights.ot.iloc[shortLateIdxs]
	
	answers = np.hstack((shortOnTimeOT, longOnTimeOT, longLateOT, shortLateOT))
	predictions = np.hstack((np.ones(shortOnTimeOT.shape), np.ones(longOnTimeOT.shape), np.zeros(longLateOT.shape), np.zeros(shortLateOT.shape)))
	
	print predictions, answers
	
	print(accuracy_score(predictions, answers))
	
	
trainAndTest()