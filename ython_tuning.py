# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 06:43:04 2018

@author: Daniel
"""

import spacy
import sklearn
import numpy
import pandas as pd
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from pprint import pprint
from time import time
import scipy
import random
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from nltk.corpus import stopwords
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint as sp_randint


random.seed(1234)

clean_data = pd.read_csv('training_cleaned.csv',header=0,encoding = "ISO-8859-1")

vectorizer = TfidfVectorizer( ngram_range = (1,2), 
                             lowercase = False,
                             stop_words = stopwords.words('english'),
                             min_df = 0.02,
                             max_df=0.98,
                             norm='l1')


#c_vectorizer = CountVectorizer(ngram_range = (1,2),
#                               lowercase=False,
#                               stop_words = stopwords.words('english'),
#                               min_df = 0.05,
#                               max_df = 0.95)


#training_tfidf_vector = vectorizer.fit_transform(clean_data['pos'].astype('U').values)
training_tfidf_nopos = vectorizer.fit_transform(clean_data['cleaned'].astype('U').values)

#training_count_vector = c_vectorizer.fit_transform(clean_data['pos'].astype('U').values)
#training_c_vector_nopos =  c_vectorizer.fit_transform(clean_data['cleaned'].astype('U').values)

ind = random.sample(range(106445), k=10000)

train_labels = pd.read_table('training_labels_final.txt',sep=" ",header=None,names=['ID','Label'])
y = train_labels.Label[ind]


#train_x = training_tfidf_vector[ind,:]
#train_c = training_count_vector[ind,:]
#train_nopos_c = training_c_vector_nopos[ind,:]
train_tfidf_nopos = training_tfidf_nopos[ind,:]


scaler = sklearn.preprocessing.MaxAbsScaler()
#X = scaler.fit_transform(train_x)
#C = scaler.fit_transform(train_c)
#NPOS = scaler.fit_transform(train_nopos_c)
tfidf_x = scaler.fit_transform(train_tfidf_nopos)

# Train classifiers
#
# For an initial search, a logarithmic grid with basis
# 10 is often helpful. Using a basis of 2, a finer
# tuning can be achieved but at a much higher cost.
#best_c = numpy.logspace(1, 10, 1)
#best_g = numpy.logspace(-1, -1, 1)
#best_k = ['sigmoid']
#param_grid = dict(gamma=best_g, C=best_c, kernel=best_k)
#
#count_pos = GridSearchCV(SVC(), param_grid=param_grid,n_jobs=-1, scoring = 'f1_macro', verbose = 5)
#count_pos.fit(C,y)
#
#print("The best parameters for count_pos are %s with a score of %0.2f"
#      % (count_pos.best_params_, count_pos.best_score_))
#
#count_npos =  GridSearchCV(SVC(), param_grid=param_grid,n_jobs=-1, scoring = 'f1_macro',verbose= 5)
#count_npos.fit(NPOS,y)
#
#print("The best parameters for count_npos are %s with a score of %0.2f"
#      % (count_npos.best_params_, count_npos.best_score_))
#
#tfidf_npos =  GridSearchCV(SVC(), param_grid=param_grid,n_jobs=-1, scoring = 'f1_macro',verbose= 5)
#tfidf_npos.fit(tfidf_x,y)
#
#print("The best parameters for count_npos are %s with a score of %0.2f"
#      % (tfidf_npos.best_params_, tfidf_npos.best_score_))



# build a classifier
clf = RandomForestClassifier(n_estimators=20)


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = numpy.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# specify parameters and distributions to sample from
param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 40),
              "min_samples_split": sp_randint(2, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search, cv=3, njobs=-1, verbose=5,scoring = 'f1_macro')

random_search.fit(tfidf_x, y)
report(random_search.cv_results_)

C_range = numpy.logspace(-2, 6, 8)
gamma_range = numpy.logspace(-4, 3, 7)
kernel = ['sigmoid']


param_grid = dict(gamma=gamma_range, C=C_range, kernel=kernel)
scoring = {'accuracy': 'accuracy','f1':'f1_macro','bal_acc':'balanced_accuracy'}

grid = GridSearchCV(SVC(), param_grid=param_grid,n_jobs=-1, scoring = 'f1_macro',verbose=5)
grid.fit(tfidf_x, y)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))



