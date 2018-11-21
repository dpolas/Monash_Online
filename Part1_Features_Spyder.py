# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 16:04:25 2018

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


## Define some functions

def read_file(filename):

    id_items = []
    text_items = []
    with open(filename,'r', encoding='utf-8') as file:
        for line in file:
            if line[0:2] == 'ID':
                id_items.append(line[3:-1])  # add the text to our list removing ID plus the space
            elif line[0:4] == 'TEXT':
                text_items.append(line[5:-1])  # add the text to our list removing TEXT plus the space
    return pd.DataFrame({'ID':id_items,'Text':text_items})# create the dataframe to return


#def tokenize_and_process(document):
#    doc = nlp(document)
#    token = []
#
#    # For each token  remove punctuation, url, email, currency & numbers.
#    # Allow any entities to remain as they are (i.e. capitilised)
#
#    for tok in doc:
#        if (not tok.is_stop and not tok.is_punct and not tok.is_bracket and not tok.is_quote
#            and not tok.is_currency and not tok.like_url and not tok.like_email and not tok.like_num
#            and tok.lemma_ != "-PRON-") and tok.ent_type == 0:
#            token.append('_'.join([tok.pos_, tok.lemma_.lower().strip()]))
#        elif tok.ent_type != 0:
#            token.append('_'.join([tok.pos_, tok.text]))  # I want feature vectors to be of a form POS_word
#
#    return token

### DO THE THINGS ###

# create training data
train_data = read_file('training_docs.txt')
train_data.head()

test_data = read_file('testing_docs_shuffle.txt')
test_data.head()

train_labels = pd.read_table('training_labels_final.txt',sep=" ",header=None,names=['ID','Label'])
train_labels.head()


print(train_data.shape)
print(train_labels.shape)
print(test_data.shape)


## Create a function to use spacy to tokenise, stopword, lemmatize and POS 
#nlp = spacy.load('en_core_web_sm-2.0.0')
#
## set up your sklearn pipeline here:
## https://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html#sphx-glr-auto-examples-model-selection-grid-search-text-feature-extraction-py
#
#
## Define a pipeline combining a text feature extractor with a simple
## classifier - check if tfidf with different parameters will fit better than a simple count vectorizer
#pipeline = Pipeline([
#    ('vect', CountVectorizer(tokenizer=tokenize_and_process,lowercase=False, ngram_range=(1, 2))),
#    ('tfidf', TfidfTransformer()),
#    ('clf', SGDClassifier()),
#])
#
## uncommenting more parameters will give better exploring power but will
## increase processing time in a combinatorial way
#parameters = {
#    'vect__max_df': (0.75, 1.0),
#    # 'vect__max_features': (None, 5000, 10000, 50000),
#    #'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
#    # 'tfidf__use_idf': (True, False),
#    'tfidf__norm': ('l1','l2'),
#    #'clf__max_iter': (5,),
#    #'clf__alpha': (0.00001, 0.000001),
#    #'clf__penalty': ('l2', 'elasticnet'),
#    # 'clf__max_iter': (10, 50, 80),
#}
#
#if __name__ == "__main__":
#    # multiprocessing requires the fork to happen in a __main__ protected
#    # block
#
#    # find the best parameters for both the feature extraction and the
#    # classifier using grid_search with cv
#    grid_search = GridSearchCV(pipeline, parameters, cv=5,
#                               n_jobs=1, verbose=1)
#
#    print("Performing grid search...")
#    print("pipeline:", [name for name, _ in pipeline.steps])
#    print("parameters:")
#    pprint(parameters)
#    t0 = time()
#    grid_search.fit(train_data.Text, train_labels.Label)
#    print("done in %0.3fs" % (time() - t0))
#    print()
#
#    print("Best score: %0.3f" % grid_search.best_score_)
#    print("Best parameters set:")
#    best_parameters = grid_search.best_estimator_.get_params()
#    for param_name in sorted(parameters.keys()):
#        print("\t%s: %r" % (param_name, best_parameters[param_name]))



### Alternative Option - Use GloVe Embeddings
#nlp_glove = spacy.load('en_core_web_md',disable=['parser','textcat'])
#
#
#docs = nlp_glove.pipe((doc for doc in train_data.Text), 
#                      batch_size=10000, 
#                      n_threads=4)
#
## for each document object created, extract the glvoe vector
#doc_glove_vectors = numpy.array([doc.vector for doc in docs])
#
#
#[doc.to_disk('/trainingdocs') for doc in docs]
#
#docs_list = list(docs)

# Now simply pass the "docs" object instead of the actual text to sklearn
# grab out all the info needed - should be much faster than before (1 iteration)

## Do a test of the proper pipeline

def feature_extractor(data, increment, glove_vectors=None):
    
    # break down the file into chunks to feed to the program to ensure it runs
    # properly
    
    # store the text data and cleaned data together
    feature_data = pd.DataFrame({'Text':data.Text,'cleaned':'','pos':''})
    
    # set parameters for the chunking
    doc_len = len(data)
    increment = increment
    
     
    #use this to keep track of the current highest increment number
    # start at 0 because the top of the index isn't included
    current_number = 0
    
    for number in range(0,doc_len,increment):
        print('number being used',number)
        # if the remaining records is < 1 increment then expand the increment
        # to include the full document lenght by changing number
        if (doc_len - number) < increment:  
            number = doc_len
        
        # load the embedding and cleaning model in chunks for spacy
        nlp_test = spacy.load('en_core_web_md',disable=['parser','textcat'])
        docs_test = nlp_test.pipe((doc for doc in feature_data.Text[current_number:number]), 
                              batch_size=5000, 
                              n_threads=4)
        
        # store the objects as a list
        test_doc_list = list(docs_test)
        
        # for each document, perform the cleaning steps, regularly updting
        # to the original pandas dataframe
        for (index,doc) in enumerate(test_doc_list):
            token = []
            pos_token = []
            for tok in doc:
                 if (not tok.is_stop and not tok.is_punct and not tok.is_bracket and not tok.is_quote
                    and not tok.is_currency and not tok.like_url and not tok.like_email and not tok.like_num
                    and tok.lemma_ != "-PRON-" and tok.pos_ != 'NUM') and tok.ent_type == 0:
                     token.append(tok.lemma_.lower().strip())
                     pos_token.append('_'.join([tok.pos_, tok.lemma_.lower().strip()]))
                 elif tok.ent_type != 0:
                    token.append(tok.text)
                    pos_token.append('_'.join([tok.pos_, tok.text]))  # I want feature vectors to be of a form POS_word
                
            feature_data.cleaned.iloc[index + current_number] = ' '.join(token)
            feature_data.pos.iloc[index + current_number] = ' '.join(pos_token)
            #glove_vectors[index + current_number] = nlp_test(' '.join(token)).vector #use for updated glove vectors
                    
        current_number = number
    return(feature_data)
    
#train_glove_vectors = numpy.empty((len(train_data),300))
testing_cleaned = feature_extractor(test_data, increment = 10000)

#save the outputs
testing_cleaned.to_csv('testing_cleaned.csv')

test_data.ID.to_csv('testing_id.csv')

#numpy.savetxt("train_glove.csv", train_glove_vectors, delimiter=",")
                


# Perform TF-IDF tokenisation
vectorizer = TfidfVectorizer( ngram_range = (1,2), 
                             lowercase = False,
                             min_df = 0.02,
                             max_df=0.98,
                             norm='l1')
testing_tfidf_vector = vectorizer.fit_transform(testing_cleaned.pos)


import scipy.sparse

numpy.savez('tfidf_features_testing',testing_tfidf_vector)


scipy.io.mmwrite('tfidf_features_testing.mtx', testing_tfidf_vector, comment='', field=None, precision=None, symmetry=None)

test_data.ID.to_csv('testing_id.csv')

