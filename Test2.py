#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ExperimentalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        print('init called')
    
    def fit(self, X, y = None):
        print('fit called')
        return(self)
    
    def transform(self, X, y = None):
        print('transform called')
        #stemmer = nltk.stem.RSLPStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        corpus = []
        for tweet in X:
          review = re.sub(r"@[A-Za-z0-9_]+", " ", tweet)
          review = re.sub('RT', ' ', review)
          review = re.sub(r"https?://[A-Za-z0-9./]+", " ", review)
          review = re.sub(r"https?", " ", review)
          review = re.sub('[^a-zA-Z]', ' ', review)
          review = review.lower()
          review = review.split()
          ps = PorterStemmer()
          review = [ps.stem(word) for word in review if not word in set(all_stopwords) if len(word) > 2]
          review = ' '.join(review)
          corpus.append(review)
        return np.array(corpus)

