#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 15:34:15 2018

@author: Jordi
"""

from sklearn.linear_model import SGDClassifier
from sklearn.base import BaseEstimator
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class Preprocessor(BaseEstimator):
    classifier = SGDClassifier() #Pour notre projet nous avons choisis comme classifier â€œSGD classifierâ€
    n_pca=1 #PCA(=Principal component analysis) permet de standardiser nos donnÃ©es
    n_skb=1 #SKB=selectKbest

def __init__(self, transformer='identity'):
    self.transformer = self

def fit(self, X, y): #permet de choisir les hyperparamÃ¨tres
    PCAPip = Pipeline([('pca',PCA()), ('clf',self.classifier)]) #enchaÃ®nement des mÃ©thodes PCA et SKB
    t = [] #sÃ©lection des hyperparamÃ¨tres en fonction de accuracy-score
    i=10
    while i <int(121500):
        t.append(i)
        i+=2
        grid_search=GridSearchCV(PCAPip,{'pca__n_components':t},verbose=3,scoring=make_scorer(accuracy_score))
        grid_search.fit(X,y)
        self.n_pca=grid_search.best_params_.get('pca__n_components')
    return self

def fit_transform(self, X, y):
    return self.fit(X,y).transform(X)

def transform(self, X, y=None): #Cette methode supprime les features avec la
    #variance la plus faible pour ensuite faire une standardization afin quâ€™elles soient plus
    #facilement utilisables
    sel = VarianceThreshold(threshold=(0.05))
    X = sel.fit_transform(X)
    pca = PCA(n_components=self.n_pca)
    X = pca.fit_transform(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X


if __name__== "__main__": #test de la classe
    # We can use this to run this file as a script and test the Preprocessor
    # Use the default input and output directories if no arguments are provided
    input_dir = "../../../public_data"
    output_dir = "../results" # Create this directory if it does not exist

    basename = 'Credit'
    D = DataManager(basename, input_dir) # Load data
    print("*** Original data ***")
    print D
    
    Prepro = Preprocessor()
 
    # Preprocess on the data and load it back into D
    D.data['X_train'] = Prepro.fit_transform(D.data['X_train'], D.data['Y_train'])
    D.data['X_valid'] = Prepro.transform(D.data['X_valid'])
    D.data['X_test'] = Prepro.transform(D.data['X_test'])
  
    # Here show something that proves that the preprocessing worked fine
    print("*** Transformed data ***")
    print D
    X_train = data.drop('target', axis=1).values