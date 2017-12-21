from sklearn.svm import SVC
import csv
from random import randint, shuffle
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import pickle

entries_required_for_single_prediction = 100
# setting = "look"
setting = "cube"
# setting = "flickering"

X_y = []

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
				X_y.append((convertToFloatList(row[1:29:2]), label)) #1 to 29 to keep only EEG sensor readings, skip 2 to skip qualities



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

X_train = []
y_train = []
X_test = []
y_test = []

shuffle(X_y)
for x_y in X_y[:int(0.8*len(X_y))]:
	X_train.append(x_y[0])
	y_train.append(x_y[1])

for x_y in X_y[int(0.8*len(X_y)):]:
	X_test.append(x_y[0])
	y_test.append(x_y[1])

# X_train = preprocessing.normalize(X_train)
# X_test = preprocessing.normalize(X_test)
# X_train = preprocessing.scale(X_train)
# X_test = preprocessing.scale(X_test)

scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

load = True
if load == False:
	clf = SVC(verbose = True)
	clf.fit(X_train, y_train)
	with open(setting + "_svc_rbf.model", "wb") as fp:   #Pickling
		pickle.dump(clf, fp)
else:
	with open(setting + "_svc_rbf.model", "rb") as fp:   #Pickling
		clf = pickle.load(fp)

print clf
# print "Model's loss =",clf.loss_

if __name__ == "__main__":
	print accuracy_score(y_test, clf.predict(X_test))

X_live_test = []

def predict():
	if len(X_live_test) >= entries_required_for_single_prediction:
		X_live_test = scaler.transform(X_live_test)
		print clf.predict(X_live_test)
		X_live_test = []

def new_data(lines):
	# print "here"
	reader = csv.reader(lines, delimiter=',', quotechar='|')
	for row in reader:
		# if '0' not in row:
			if row is not None:
				new_x = convertToFloatList(row[1:29:2])
				X_live_test.append(new_x)
	# print "Length is", len(X_live_test)
	predict()


