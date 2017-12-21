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
from triangle import show
import signal

entries_required_for_single_prediction = 1
# setting = "look" #94% by ANN, 30 iter
# setting = "cube" #96% by ANN, 30 iter
setting = "flickering" #91% by ANN, 30 iter
ANN = True
SVM = False
bagging_svm = False
adaboost_decision_tree = False

enable_DWT = True
size_DWT_set = 1200


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
	# print "Performing DWT"
	sys.stdout.flush()
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

# def doDWTSet(X):
# 	X = np.array(X)
# 	if enable_DWT:
# 		X_output = np.array([])
# 		for i in range(0, len(X), size_DWT_set):
# 			X_mod = X[i:i + size_DWT_set,:].T
# 			n_cols = doDWT(X_mod[0, :], 'sym5').shape[0]
# 			X_temp = np.zeros((X_mod.shape[0], n_cols))
# 			for row in range(X_mod.shape[0]):
# 				X_temp[row, :] = doDWT(X_mod[row, :], 'sym5')
# 			if X_output.size != 0:
# 				X_output = np.append(X_output, X_temp.T, axis = 0)
# 			else:
# 				X_output = X_temp.T
# 		return X_output
# 	else:
# 		return X

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
		addDataFromFile(filename, label)
	file_regex = 'dataset2/'+ setting + '/left/*.csv'
	label = 'L'
	for filename in glob.iglob(file_regex):
		addDataFromFile(filename, label)
	file_regex = 'dataset2/'+ setting + '/up/*.csv'
	label = 'U'
	for filename in glob.iglob(file_regex):
		addDataFromFile(filename, label)
	file_regex = 'dataset2/'+ setting + '/down/*.csv'
	label = 'D'
	for filename in glob.iglob(file_regex):
		addDataFromFile(filename, label)


# loadData() #DWT performed during load itself. A switch variable to enable/disable DWT is at code start

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle = True)

# ica = FastICA()
# ica.fit(np.asarray(X_train))
# X_train = ica.transform(X_train)

# pca = PCA()
# pca.fit(X_train)
# X_train = pca.transform(X_train)

# scaler = preprocessing.StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)


if ANN:
	clf = MLPClassifier(solver = 'adam', activation = 'tanh', alpha=1e-5, hidden_layer_sizes=(50, 50), random_state=1, max_iter = 400, verbose = True, tol = 1e-6, learning_rate = 'adaptive')
elif SVM:
	clf = SVC(verbose = True)
elif bagging_svm:
	clf = BaggingClassifier(SVC(verbose = True), verbose = True, n_jobs = 3, max_features = 0.5, max_samples= 0.5)
elif adaboost_decision_tree:
	clf = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=40)

print clf


def pretty_print(freq):
	import operator
	sorted_freq = sorted(freq.items(), key=operator.itemgetter(1), reverse = True)
	return sorted_freq[0][0]

X_live_test = []
scaler = preprocessing.StandardScaler()

def predict():
	global X_live_test
	try:
		if len(X_live_test) >= entries_required_for_single_prediction:
			X_live_test = doDWTSet(X_live_test)
			X_live_test = scaler.transform(X_live_test)
			# print X_live_test
			freq = {}
			for i in clf.predict(X_live_test):
				if i in freq:
					freq[i] += 1
				else:
					freq[i] = 1
			print pretty_print(freq)
			# print freq
			X_live_test = []
		else:
			print "length too short to predict"
	except Exception:
		print traceback.print_exc()
		print sys.exc_info()[0]

parsed_till_line_number = 0
retrained = False

def is_file_okay(thread_name):
	if file is None:
		print thread_name,'File not initialized yet.'
		return False
	if isfile(file) == False:
		print thread_name,'File not created yet.'
		return False
	return True

def extract_new_data():
	global parsed_till_line_number, X_live_test
	reader = csv.reader(open(file), delimiter=',', quotechar='|')
	next(reader, None)	#ignore header line
	line_number = 0
	for row in reader:
		line_number += 1
		if line_number > parsed_till_line_number:
			if '0' not in row:	#only if quality is not zero
				if row is not None:
					X_live_test.append(convertToFloatList(row[1:29:2]))
	# print "Live test data length is", len(X_live_test), parsed_till_line_number, line_number
	parsed_till_line_number = line_number
	sys.stdout.flush()

def parse_new_data(should_predict = True):
	
	while True:
		if is_file_okay('Prediction Thread') == False:
			sleep(5)
			continue
		if retrained == False:
			print 'Not retrained. Can\'t predict. Going back to sleep for 20'
			sleep(20)
			continue
		extract_new_data()
		# print X_live_test
		if len(X_live_test) >= entries_required_for_single_prediction:
			predict()
		else:
			print ".",
		sleep(10)



file = None
def init(file_path):
	global file
	file = file_path
	print "Prediction thread: Path set as",file

# X = np.array([])
# y = []

def retrain_for_label(label):
	global X_live_test, clf,X,y
	while is_file_okay('Training Thread') == False:
		sleep(5)
	print "See",label
	sleep(7)
	print "GOOOO ",label
	sleep(3)
	extract_new_data()
	X_live_test = []
	sleep(50)
	# show()
	print "Fetching new data",
	extract_new_data()
	print "Data extracted. # samples",len(X_live_test)
	y_live_test = []
	# X_live_test = X_live_test[X_live_test.shape[0] - size_DWT_set:,:]
	X_live_test = doDWTSet(X_live_test)
	print X_live_test
	while len(y_live_test) != X_live_test.shape[0]:
		y_live_test.append(label)
	print "Training with new data",
	sys.stdout.flush()
	print X_live_test.shape, len(y_live_test)
	print X_live_test[0]
	print "Trained"
	if X.size != 0:
		X = np.append(X, doDWTSet(X_live_test), axis = 0)
	else:
		X = doDWTSet(X_live_test)
	while len(y) != X.shape[0]:
		y.append(label)
	X_live_test = []

def retrain():
	global retrained,X,y,clf,scaler
	load = True
	if load:
		# with open("live_model_4class_r_bfre.model", "rb") as fp:   #Pickling
		with open("live_model_4class_r_bre.model", "rb") as fp:   #Pickling
			clf = pickle.load(fp)
		with open("live_scaler_4class_r_bre.model", "rb") as fp:   #Pickling
			scaler = pickle.load(fp)
	else:
		retrain_for_label('Eyebrow Raised')
		# retrain_for_label('Focus')
		retrain_for_label('Blinking')
		retrain_for_label('Relaxed')
		# retrain_for_label('Smile')
		# retrain_for_label('F')
		# retrain_for_label('N')
		scaler.fit(X)
		X = scaler.transform(X)
		clf.fit(X, y)
		with open("live_model_4_r.model", "wb") as fp1:   #Pickling
			pickle.dump(clf, fp1)
		with open("live_scaler_4_r.model", "wb") as fp1:   #Pickling
			pickle.dump(scaler, fp1)
	retrained = True
	print "Started prediction thread"
	thread = Thread(target = parse_new_data)
	thread.start()



if __name__ == "__main__":
	# X_test = ica.transform(X_test)
	# X_test = pca.transform(X_test)
	X_test = scaler.transform(X_test)
	print "Accuracy:",accuracy_score(y_test, clf.predict(X_test))
else:
	thread = Thread(target = retrain)
	thread.start()
	print "Started training thread"
	
