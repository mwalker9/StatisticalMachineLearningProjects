from bottle import route, run, request, install
from bottle_sqlite import SQLitePlugin
import json
from sklearn.svm import SVR
import numpy as np

class state: 
	model = SVR()
	cols = []
	trained = False

curState = state()

install(SQLitePlugin(dbfile='./testDB.sqlite'))

@route('/getTables', method='GET')
def getTables(db):
	sql = "select name from sqlite_master where type='table'"
	c = db.execute(sql)
	tbls = []
	for row in c:
		tbls.append(row[0])
	return json.dumps(tbls)

@route('/getColumns', method='POST')
def getColumns(db):
	reqJson = request.json
	tblName = reqJson["tableName"]
	sql = "PRAGMA table_info(" + tblName + ")"
	c = db.execute(sql)
	cols = []
	for row in c:
		cols.append(row[1])
	return json.dumps(cols)
	
@route('/getPrimaryKey', method='POST')
def getPrimaryKey(db):
	reqJson = request.json
	tblName = reqJson["tableName"]
	sql = "PRAGMA table_info(" + tblName + ")"
	c = db.execute(sql)
	for row in c:
		if row[5] == 1:
			return row[1]
	return ''

@route('/getForeignKeys', method='POST')
def getForeignKey(db):
	reqJson = request.json
	tblName = reqJson["tableName"]
	sql = "PRAGMA foreign_key_list(" + tblName + ")"
	c = db.execute(sql)
	fks = []
	for row in c:
		fks.append(row[3])
	return json.dumps(fks)

@route('/train', method='PUT')
def train(db):
	curState.trained = False
	curState.model = SVR()
	reqJson = request.json
	inputs = reqJson["inputs"]
	outputs = reqJson["output"]
	keyMappingCls = []
	if reqJson.has_key("keyMapping"):
		keyMappingArr = reqJson["keyMapping"]
		for mapEntry in keyMappingArr:
			keyMappingCls.append(mapEntry.keys()[0] + "." + mapEntry.values()[0])
	tbls = []
	cols = []
	curState.cols = cols
	for obj in inputs:
		tblName = obj.keys()[0]
		tbls.append(tblName)
		val = obj.values()[0]
		for colName in val:
			cols.append(tblName + "." + colName)
	outputTbl = outputs.keys()[0]
	outputCol = outputTbl + "." + outputs.values()[0]
	sql = "SELECT ";
	for col in cols:
		sql += col + ", "
	sql += outputCol
	sql += " FROM "
	for tbl in tbls[:-1]:
		sql += tbl + ", "
	if outputTbl not in tbls:
		sql += outputTbl + ", "
	sql += tbls[-1]
	sql += " WHERE 1=1 "
	for i, val in enumerate(keyMappingCls):
		if i + 1 == len(keyMappingCls):
			break
		sql += "and " + val + " = " + keyMappingCls[i+1] + " " 
	trainingSet = []
	trainingOutputs = []
	c = db.execute(sql)
	for row in c:
		outputIdx = len(row) - 1		
		trainingRow = []
		for i, val in enumerate(row):
			if i == outputIdx:
				trainingOutputs.append(val)
			else:
				trainingRow.append(val)
		trainingSet.append(trainingRow)
	curState.model.fit(trainingSet, trainingOutputs)
	curState.trained = True
	return "Success"
		

@route('/predict', method='POST')
def predict(db):
	reqJson = request.json
	inputs = reqJson["inputs"]
	valueMap = {}
	for tblMap in inputs:
		tblName = tblMap.keys()[0]
		colArray = tblMap.values()[0]
		for colMap in colArray:
			colName = colMap.keys()[0]
			value = float(colMap.values()[0])
			key = tblName + "." + colName
			key = key.encode('ascii','ignore')
			valueMap[key] = value
	if not curState.trained:
		return "waiting"
	instance = np.zeros(len(curState.cols))
	for keyName in valueMap.keys():
		idx = curState.cols.index(keyName)
		instance[idx] = valueMap[keyName]
	return str(curState.model.predict(np.atleast_2d(instance))[0])

run(host='localhost', port=8080, debug=False)