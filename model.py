'''
Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
'''
import imp, pickle, numpy as np
from sklearn.base import BaseEstimator
from os.path import isfile
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from preprocessing import Preprocessor


class model(BaseEstimator):
    def __init__(self):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation. 
        '''
        self.num_train_samples = 121499
        self.num_feat = 56     #attributs
        self.num_labels = 1   #classes
        self.is_trained = False
        self.clf = Pipeline([('preprocessor', Preprocessor()), ('class', SGDClassifier())])

    def fit(self, X, y):
        '''
        This function should train the model parameters.
        Here we do nothing in this example...
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        For classification, labels could be either numbers 0, 1, ... c-1 for c classe
        or one-hot encoded vector of zeros, with a 1 at the kth position for class k.
        The AutoML format support on-hot encoding, which also works for multi-labels problems.
        Use data_converter.convert_to_num() to convert to the category number format.
        For regression, labels are continuous values.
        '''
        return self.clf.fit(X, y)
        
    def predict(self, X, y):
        '''
        This function should provide predictions of labels on (test) data.
        Here we just return zeros...
        Make sure that the predicted values are in the correct format for the scoring
        metric. For example, binary classification problems often expect predictions
        in the form of a discriminant value (if the area under the ROC curve it the metric)
        rather that predictions of the class labels themselves. For multi-class or multi-labels
        problems, class probabilities are often expected if the metric is cross-entropy.
        Scikit-learn also has a function predict-proba, we do not require it.
        The function predict eventually can return probabilities.
        '''
        return self.clf.predict(X)

    def score(self, X, y):
        '''
        This function returns the score of the model by using the method
        score of the SGDClassifier
        X : training data
        y : target values
        '''
        return self.clf.score(X,y)

    def cross_val_score(self, X, y):
        return cross_val_score(self.clf, X, y, cv=5)
    
    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile) as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self

if __name__=='__main__':
    # Find the files containing corresponding data
    # To find these files successfully:
    # you should execute this "model.py" script in the folder "sample_code_submission"
    # and the folder "public_data" should be in the SAME folder as the starting kit
    path_to_training_data = "../../../public_data/credit_train.data"
    path_to_training_label = "../../../public_data/credit_train.solution"
    path_to_testing_data = "../../../public_data/credit_test.data"
    path_to_validation_data = "../../../public_data/credit_valid.data"

    # Find the program computing AUC metric
    path_to_metric = "../scoring_program/my_metric.py"
    auc_metric = imp.load_source('metric', path_to_metric).auc_metric_

    # use numpy to load data
    X_train = np.loadtxt(path_to_training_data)
    y_train = np.loadtxt(path_to_training_label)
    X_test = np.loadtxt(path_to_testing_data)
    X_valid = np.loadtxt(path_to_validation_data)


    # TRAINING ERROR
    # generate an instance of our model (clf for classifier)
    clf = model()
    # train the model
    clf.fit(X_train, y_train)
    # to compute training error, first make predictions on training set
    y_hat_train = clf.predict(X_train, y_train)
    # then compare our prediction with true labels using the metric
    training_error = auc_metric(y_train, y_hat_train)

    score = clf.cross_val_score(X_train, y_train)

    print "Training: ", training_error
    print "Score cross-validation: ", score
    
							  
    
