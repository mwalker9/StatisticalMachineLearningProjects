import matplotlib.pyplot as plt
import seaborn
import pandas
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, accuracy_score

def getRating(row, ratingsMatrix, movieMap, userMap, idx, predictedRatings):
	predictedRatings[0, idx] = row["testID"]
	try:
		movieIdx = np.where(movieMap == row["movieID"])[0][0]
	except IndexError:
		predictedRatings[1, idx] = 3.75
		return
	unknown = np.where(ratingsMatrix[:, movieIdx] == 3.7123)
	userIdx = np.where(userMap == row["userID"])[0][0]
	specificMovieRatings = ratingsMatrix[:, movieIdx]
	ratingsMatrix = np.delete(ratingsMatrix, movieIdx, axis=1)
	currentUser = ratingsMatrix[userIdx, :]
	ratingsMatrix = np.delete(ratingsMatrix, unknown, axis=0)
	if ratingsMatrix.shape[0] == 0:
		rating = movieInfo.loc[movieInfo['id'] == row["movieID"]].rtAudienceRating.values[0]
		if rating == 0:
			rating = .5
		#rating = round(rating * 2) / 2.
		if rating < .5:
			rating = .5
		elif rating > 5:
			rating = 5
		predictedRatings[1, idx] = rating
		return
	specificMovieRatings = np.delete(specificMovieRatings, unknown, axis=0)
	model.fit(ratingsMatrix)
	k = min(ratingsMatrix.shape[0], 9)
	dist, indeces = model.kneighbors(currentUser.reshape(1, -1), n_neighbors = k)
	indeces = indeces[0]
	allRatings = specificMovieRatings[indeces]
	dist = dist[0]
	dist = [1./d for d in dist]
	dist = dist/sum(dist)
	rating = np.dot(dist, allRatings)
	#rating = round(rating * 2) / 2.
	if rating < .5:
		rating = .5
	elif rating > 5:
		rating = 5
	predictedRatings[1, idx] = rating
	return
	
ur = pandas.read_csv('.\\movie_training_data\\user_ratedmovies_train.dat','\t')

movieInfo = pandas.read_csv(".\\movie_training_data\\movies.dat", '\t')
movieInfo.rtAudienceRating = movieInfo.rtAudienceRating.replace("\\N", 3.5)
movieInfo.rtAudienceRating = pandas.to_numeric(movieInfo.rtAudienceRating)


ur = ur[["userID" , "movieID", "rating"]] 

# create a test/train split

testInstances = 85000
 
ur_test = pandas.read_csv("predictions.dat", '\t')
ur_train = ur

max_movie_id = len(np.unique(ur["movieID"]))
max_user_id = len(np.unique(ur["userID"]))

movieMap = np.unique(ur["movieID"])
userMap = np.unique(ur["userID"])

ratingsMatrix = np.ones((max_user_id, max_movie_id)) * 3.7123

for index, row in ur_train.iterrows():
	movieIdx = np.where(movieMap == row["movieID"])
	userIdx = np.where(userMap == row["userID"])
	ratingsMatrix[userIdx, movieIdx] = row["rating"]

predictedRatings = np.zeros((2, testInstances))
	
model = NearestNeighbors()
	
idx = 0
	
for index, row in ur_test.iterrows():
	getRating(row, ratingsMatrix, movieMap, userMap, idx, predictedRatings)
	idx += 1
	print(idx, idx/float(testInstances))

df = pandas.DataFrame(data=predictedRatings.T, columns=["testID", "predicted rating"])
df[['testID']] = df[['testID']].astype(int)
df.to_csv("mypredictions.dat", index=False)