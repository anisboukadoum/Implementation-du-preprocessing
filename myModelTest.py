# -*- coding: utf-8 -*-
"""

@author: mamadou.ndiaye
"""

from myData_manager import DataManager
from myModel import Model
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cross_validation import cross_val_score

input_dir = "../public_data"
output_dir = "../res"
basename = 'credit'
D = DataManager(basename, input_dir)
print D
myClassifier = Model()
Ytrue_train = D.data['Y_train']
myClassifier.fit(D.data['X_train'], Ytrue_train)
Ypredict_train = myClassifier.predict(D.data['X_train'])
Ypredict_proba_train = myClassifier.predict_proba(D.data['X_train'])
Ypredict_valid = myClassifier.predict(D.data['X_valid'])
Ypredict_test = myClassifier.predict(D.data['X_test'])
accuracy_train = accuracy_score(Ytrue_train, Ypredict_train)
accuracy_crossval = cross_val_score(myClassifier, D.data['X_train'], Ytrue_train, cv=5, scoring='accuracy')
print "Training Accuracy = %5.2f" % (accuracy_train)
print "Cross-validation Accuracy = %5.2f +-%5.2f" % (accuracy_crossval.mean(), accuracy_crossval.std())
print confusion_matrix(Ytrue_train, Ypredict_train)
