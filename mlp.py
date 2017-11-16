from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
import csv
import numpy as np
from random import randint, shuffle
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import FastICA, PCA
import matplotlib.pyplot as plt
import pywt
import pywt.data
import pickle
import sys
import traceback
from threading import Thread
from time import sleep
from os.path import isfile
import glob

entries_required_for_single_prediction = 1
# setting = "look" #94% by ANN, 30 iter
# setting = "cube" #96% by ANN, 30 iter
setting = "flickering" #91% by ANN, 30 iter
ANN = True
SVM = False
bagging_svm = False
adaboost_decision_tree = False

enable_DWT = False

X = np.array([])
y = []

assert(ANN or SVM or bagging_svm or adaboost_decision_tree) #Either ANN or SVM or bagging_svm or adaboost enabled
assert(not( (ANN and SVM) or (ANN and bagging_svm) or  (bagging_svm and SVM) or (ANN and adaboost_decision_tree) or (adaboost_decision_tree and SVM) or (adaboost_decision_tree and bagging_svm)))

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
	if enable_DWT:
		X = X.T
		n_cols = doDWT(X[0, :], 'sym5').shape[0]
		X_temp = np.zeros((X.shape[0], n_cols))
		for row in range(X.shape[0]):
			X_temp[row, :] = doDWT(X[row, :], 'sym5')
		X = X_temp.T
		return X
	else:
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
				local_X.append(convertToFloatList(row[1:29:2]))
				y.append(label)
	if X.size != 0:
		X = np.append(X, doDWTSet(local_X), axis = 0)
	else:
		X = doDWTSet(local_X)
	while len(y) != X.shape[0]:
		y.append(label)



def loadData():
	file_regex = 'dataset2/'+ setting + '/right/*.csv'
	label = 'R'
	for filename in glob.iglob(file_regex):
		print(filename)
		addDataFromFile(filename, label)
	file_regex = 'dataset2/'+ setting + '/left/*.csv'
	label = 'L'
	for filename in glob.iglob(file_regex):
		print(filename)
		addDataFromFile(filename, label)
	file_regex = 'dataset2/'+ setting + '/up/*.csv'
	label = 'U'
	for filename in glob.iglob(file_regex):
		print(filename)
		addDataFromFile(filename, label)
	file_regex = 'dataset2/'+ setting + '/down/*.csv'
	label = 'D'
	for filename in glob.iglob(file_regex):
		print(filename)
		addDataFromFile(filename, label)


loadData() #DWT performed during load itself. A switch variable to enable/disable DWT is at code start

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ica = FastICA()
# ica.fit(np.asarray(X_train))
# X_train = ica.transform(X_train)

# pca = PCA()
# pca.fit(X_train)
# X_train = pca.transform(X_train)

scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)


load = False
if load == False:
	if ANN:
		clf = MLPClassifier(solver = 'adam', activation = 'tanh', alpha=1e-5, hidden_layer_sizes=(50, 50), random_state=1, max_iter = 300, verbose = True, tol = 1e-6, learning_rate = 'adaptive')
		clf.fit(X_train, y_train)
		with open(setting + "_neural_adam_tanh_50_50_new.model", "wb") as fp:   #Pickling
			pickle.dump(clf, fp)
	elif SVM:
		clf = SVC(verbose = True)
		clf.fit(X_train, y_train)
		with open(setting + "_svc_rbf_new.model", "wb") as fp:   #Pickling
			pickle.dump(clf, fp)
	elif bagging_svm:
		clf = BaggingClassifier(SVC(verbose = True), verbose = True, n_jobs = 3, max_features = 0.5, max_samples= 0.5)
		clf.fit(X_train, y_train)
		with open(setting + "_bagging_svc_rbf_new.model", "wb") as fp:   #Pickling
			pickle.dump(clf, fp)
	elif adaboost_decision_tree:
		clf = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=40)
		clf.fit(X_train, y_train)
		with open(setting + "_adaboost_decision_tree_new.model", "wb") as fp:   #Pickling
			pickle.dump(clf, fp)
else:
	if ANN:
		with open(setting + "_neural_adam_tanh_50_50.model", "rb") as fp:   #Pickling
			clf = pickle.load(fp)
	elif SVM:
		with open(setting + "_svc_rbf.model", "rb") as fp:   #Pickling
			clf = pickle.load(fp)
	elif bagging_svm:
		with open(setting + "_bagging_svc_rbf_new.model", "rb") as fp:   #Pickling
			clf = pickle.load(fp)
	elif adaboost_decision_tree:
		with open(setting + "_adaboost_decision_tree.model", "rb") as fp:   #Pickling
			clf = pickle.load(fp)

print clf


X_live_test = []

def predict():
	global X_live_test
	try:
		if len(X_live_test) >= entries_required_for_single_prediction:
			X_live_test = doDWTSet(X_live_test)
			X_live_test = scaler.transform(X_live_test)
			freq = {}
			for i in clf.predict(X_live_test):
				if i in freq:
					freq[i] += 1
				else:
					freq[i] = 1
			print freq
			X_live_test = []
		else:
			print "length too short to predict"
	except Exception:
		print traceback.print_exc()
		print sys.exc_info()[0]

parsed_till_line_number = 0

def parse_new_data():
	global parsed_till_line_number
	while True:
		if file is None:
			print 'File not initialized yet. Can\'t predict. Going back to sleep'
			sleep(5)
			continue
		if isfile(file) == False:
			print 'File not created yet. Can\'t predict. Going back to sleep'
			sleep(5)
			continue
		reader = csv.reader(open(file), delimiter=',', quotechar='|')
		next(reader, None)
		line_number = 0
		for row in reader:
			line_number += 1
			if line_number > parsed_till_line_number:
				if '0' not in row:
					if row is not None:
						# print convertToFloatList(row[1:29:2])
						X_live_test.append(convertToFloatList(row[1:29:2]))
		parsed_till_line_number = line_number
		sys.stdout.flush()
		if len(X_live_test) >= entries_required_for_single_prediction:
			print "Received length is", len(X_live_test)
			predict()
		else:
			print ".",
		sleep(10)



file = None
def init(file_path):
	global file
	file = file_path
	print "Prediction thread: Path set as",file


if __name__ == "__main__":
	# X_test = ica.transform(X_test)
	# X_test = pca.transform(X_test)
	X_test = scaler.transform(X_test)
	print "Accuracy:",accuracy_score(y_test, clf.predict(X_test))
else:
	thread = Thread(target = parse_new_data)
	thread.start()
	print "Started prediction thread"
