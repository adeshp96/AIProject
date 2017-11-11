from sklearn.neural_network import MLPClassifier
import csv
import numpy as np
from random import randint, shuffle
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
import pywt
import pywt.data
import pickle
import sys
import traceback

entries_required_for_single_prediction = 1
setting = "look"
# setting = "cube"
# setting = "flickering"

X = np.array([])
y = []

def doDWT(data, w):
	mode = pywt.Modes.smooth
	w = pywt.Wavelet(w)
	a = data
	cd = []
	for i in range(8):
		(a, d) = pywt.dwt(a, w, mode)
		cd.append(d)
	rec_d = []
	for i, coeff in enumerate(cd):
		coeff_list = [None, coeff] + [None] * i
		rec_d.append(pywt.waverec(coeff_list, w))
	return rec_d[-1]


def doDWTSet(X):
	X = np.array(X)
	X = X.T
	n_cols = doDWT(X[0, :], 'sym5').shape[0]
	X_temp = np.zeros((X.shape[0], n_cols))
	for row in range(X.shape[0]):
		X_temp[row, :] = doDWT(X[row, :], 'sym5')
	X = X_temp.T
	return X

def convertToFloatList(l):
	l = [float(i) for i in l]
	return l

def addDataFromFile(file, label):
	global X
	print file
	local_X = []
	with open(file) as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='|')
		next(reader, None)
		for row in reader:
			if '0' not in row:
				# X_y.append((convertToFloatList(row[1:29:2]), label)) #1 to 29 to keep only EEG sensor readings, skip 2 to skip qualities
				local_X.append(convertToFloatList(row[1:29:2]))
				y.append(label)
	# print X.shape, doDWTSet(local_X).shape 
	if X.size != 0:
		X = np.append(X, doDWTSet(local_X), axis = 0)
	else:
		X = doDWTSet(local_X)
	while len(y) != X.shape[0]:
		y.append(label)
		# print "y's size",len(y)



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

# ica = FastICA()
# ica.fit(np.asarray(X_train))
# X_train = ica.transform(X_train)


scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)




load = False
if load == False:
	# clf = MLPClassifier(solver = 'sgd', activation = 'tanh', alpha=1e-5, hidden_layer_sizes=(50, 50), random_state=1, max_iter = 300, verbose = True, tol = 1e-6, learning_rate = 'constant', learning_rate_init = 0.01)
	# clf = MLPClassifier(solver = 'adam', activation = 'tanh', alpha=1e-5, hidden_layer_sizes=(50, 50), random_state=1, max_iter = 300, verbose = True, tol = 1e-6, learning_rate = 'adaptive', learning_rate_init = 0.01)
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
	# X_test = ica.transform(X_test)
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


