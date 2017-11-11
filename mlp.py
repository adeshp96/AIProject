from sklearn.neural_network import MLPClassifier
import csv
import numpy as np
from random import randint, shuffle
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pickle
import sys
import traceback

entries_required_for_single_prediction = 1
setting = "look"
# setting = "cube"
# setting = "flickering"

X = []
y = []

def convertToFloatList(l):
	l = [float(i) for i in l]
	return l

def addDataFromFile(file, label):
	print file
	with open(file) as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='|')
		next(reader, None)
		for row in reader:
			if '0' not in row:
				# X_y.append((convertToFloatList(row[1:29:2]), label)) #1 to 29 to keep only EEG sensor readings, skip 2 to skip qualities
				X.append(convertToFloatList(row[1:29:2]))
				y.append(label)



def loadData():
	addDataFromFile('dataset/'+ setting + '/adesh/left.csv', "L")
	addDataFromFile('dataset/'+ setting + '/adesh/down.csv', "D")
	addDataFromFile('dataset/'+ setting + '/adesh/up.csv', "U")
	addDataFromFile('dataset/'+ setting + '/adesh/right.csv', "R")
	addDataFromFile('dataset/'+ setting + '/kunal/left.csv', "L")
	addDataFromFile('dataset/'+ setting + '/kunal/down.csv', "D")
	addDataFromFile('dataset/'+ setting + '/kunal/up.csv', "U")
	addDataFromFile('dataset/'+ setting + '/kunal/right.csv', "R")
	addDataFromFile('dataset/'+ setting + '/rishabh/left.csv', "L")
	addDataFromFile('dataset/'+ setting + '/rishabh/down.csv', "D")
	addDataFromFile('dataset/'+ setting + '/rishabh/up.csv', "U")
	addDataFromFile('dataset/'+ setting + '/rishabh/right.csv', "R")


loadData()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

load = True
if load == False:
	clf = MLPClassifier(solver = 'adam', activation = 'tanh', alpha=1e-5, hidden_layer_sizes=(50, 50), random_state=1, max_iter = 300, verbose = True, tol = 1e-6, learning_rate = 'adaptive')
	clf.fit(X_train, y_train)
	with open(setting + "_neural_adam_tanh_50_50_new.model", "wb") as fp:   #Pickling
		pickle.dump(clf, fp)
else:
	with open(setting + "_neural_adam_tanh_50_50.model", "rb") as fp:   #Pickling
		clf = pickle.load(fp)

print clf
print "Model's loss =",clf.loss_

if __name__ == "__main__":
	X_test = scaler.transform(X_test)
	print accuracy_score(y_test, clf.predict(X_test))

X_live_test = []

def predict():
	global X_live_test
	try:
		if len(X_live_test) >= entries_required_for_single_prediction:
			X_live_test = scaler.transform(X_live_test)
			print clf.predict(X_live_test)
			X_live_test = []
		else:
			print "length too short to predict"
	except Exception:
		print traceback.print_exc()
		print sys.exc_info()[0]

def new_data(lines):
	reader = csv.reader(lines, delimiter=',', quotechar='|')
	for row in reader:
		if '0' not in row:
			if row is not None:
				new_x = convertToFloatList(row[1:29:2])
				X_live_test.append(new_x)
	print "Received length is", len(X_live_test)
	sys.stdout.flush()
	predict()


