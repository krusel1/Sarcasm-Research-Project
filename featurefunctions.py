# -*- coding: utf-8 -*-
"""FeatureFunctions.ipynb

Automatically generated by Colaboratory.

"""


import pandas as pd
import numpy as np
import os
import operator 
import re
import sys
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.stem import PorterStemmer
from collections import Counter
import string
from scipy.sparse import hstack
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

#Preproccessing


# helper function for pre-processing text given a file
def process_file(text):
    #put text in all lower case letters 
    #all_text = text.lower()
    all_text = text

    #remove all non-alphanumeric chars
    #all_text = re.sub(r"[^a-zA-Z0-9]", " ", all_text)
    #remove newlines/tabs, etc. so it's easier to match phrases, later
    all_text = re.sub(r"\t", " ", all_text)
    all_text = re.sub(r"\n", " ", all_text)
    all_text = re.sub("  ", " ", all_text)
    all_text = re.sub("   ", " ", all_text)
    all_text = re.sub(r'(@USER)', ' ', all_text)
    return all_text

def create_text_column(df):
  #create copy to modify
  text_df = df.copy()
    
  #store processed text
  text = []
    
      # for each file (row) in the df, read in the file 
  for row_i in df.index:
      filename = df.iloc[row_i]['response']
      file_text = process_file(str(filename))
          #append processed text to list
      text.append(file_text)
    
    #add column to the copied dataframe
  text_df['text'] = text
    
  return text_df

from nltk.sentiment.vader import SentimentIntensityAnalyzer  
  
  #Sentiment Analyzer VADER
def nltk_sentiment(sentence):
  nltk_sentiment = SentimentIntensityAnalyzer()
  score = nltk_sentiment.polarity_scores(str(sentence))
  return score

  pos_family = {
      'noun' : ['NN','NNS','NNP','NNPS'],
      'pron' : ['PRP','PRP$','WP','WP$'],
      'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
      'adj' :  ['JJ','JJR','JJS'],
      'adv' : ['RB','RBR','RBS','WRB']
  }

  from textblob import TextBlob
  # function to check and get the part of speech tag count of a words in a given sentence
def check_pos_tag(x, flag):
    pos_family = {
      'noun' : ['NN','NNS','NNP','NNPS'],
      'pron' : ['PRP','PRP$','WP','WP$'],
      'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
      'adj' :  ['JJ','JJR','JJS'],
      'adv' : ['RB','RBR','RBS','WRB']
  }
  
    cnt = 0
    try:
        wiki = textblob.TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
    except:
        pass
    return cnt

 

def features(df1, df2):
  # creating lists to keep pos, neu, neg, and compound scores --- later to be used to create a dataframe

  complete_df = df1
  features_df = df2
  vs_compound = []
  vs_pos = []
  vs_neu = []
  vs_neg = []

# extracting vader scores for each entry in the data
# (we're not using context yet.)
# note that the compound score is rescaled to the [0,1] range
# some classiifers don't take negative values (e.g., MultinomialNB)

  for row in complete_df["response"]:
      score = nltk_sentiment(str(row))
  
      neg = float(score['neg'])
      vs_neg.append(neg)
      neu =float(score['neu'])
      vs_neu.append(neu)
      pos =float(score['pos'])
      vs_pos.append(pos)
      compound = float((score['compound']+1)/2) # rescaling to the [0,1] range
      vs_compound.append(compound)
        

##Create feature vector 
  features_df["vader_pos"] = vs_pos
  features_df["vader_neg"] = vs_neg
  features_df["vader_neu"] = vs_neu
  features_df["vader_compound"] = vs_compound

  features_df['noun_count'] = complete_df['response'].apply(lambda x: check_pos_tag(x, 'noun')).astype(float)
  features_df['verb_count'] = complete_df['response'].apply(lambda x: check_pos_tag(x, 'verb')).astype(float)
  features_df['adj_count'] = complete_df['response'].apply(lambda x: check_pos_tag(x, 'adj')).astype(float)
  features_df['adv_count'] = complete_df['response'].apply(lambda x: check_pos_tag(x, 'adv')).astype(float)
  features_df['pron_count'] = complete_df['response'].apply(lambda x: check_pos_tag(x, 'pron')).astype(float)

  features_df['char_count'] = complete_df['response'].apply(len).astype(float)
  features_df['word_count'] = complete_df['response'].apply(lambda x: len(x.split())).astype(float)
  features_df['word_density'] = features_df['char_count']/ (features_df['word_count']+1).astype(float)
  features_df['punctuation_count'] = complete_df['response'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))).astype(float)
  features_df['upper_case_word_count'] = complete_df['response'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()])).astype(float)

  return features_df

def normalized_features(df1,df2):
  norm_features_df = df1
  features_df = df2


  norm_features_df['noun_count_percent'] = features_df['noun_count']/ (features_df['word_count']+1).astype(float)
  norm_features_df['noun_count_percent'] = features_df['verb_count']/ (features_df['word_count']+1).astype(float)
  norm_features_df['adj_count_percent'] = features_df['adj_count']/ (features_df['word_count']+1).astype(float)
  norm_features_df['adv_count_percent'] = features_df['adv_count']/ (features_df['word_count']+1).astype(float)
  norm_features_df['pro_count_percent'] = features_df['pron_count']/ (features_df['word_count']+1).astype(float)

  norm_features_df['char_count'] = features_df['char_count']/ (features_df['word_count']+1).astype(float)
  norm_features_df['word_density'] = features_df['word_density']/ (features_df['word_count']+1).astype(float)
  norm_features_df['punctuation_count'] = features_df['punctuation_count']/ (features_df['word_count']+1).astype(float)
  norm_features_df['upper_case_word_count'] = features_df['upper_case_word_count']/ (features_df['word_count']+1).astype(float)


  norm_features_df["vader_pos"] = features_df["vader_pos"]
  norm_features_df["vader_neg"] = features_df["vader_neg"]
  norm_features_df["vader_neu"] = features_df["vader_neu"]
  norm_features_df["vader_compound"] = features_df["vader_compound"]

   

  return norm_features_df



from numpy import array
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix

def count_features(df):



  count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}',max_features= 3000) # word that appears 1 or more times
  count_vect.fit(df['response'])

  # to show resulting vocabulary; the numbers are not counts, they are the position in the sparse vector.
  count_vect.vocabulary_

  # transform the training and validation data using count vectorizer object: doc x term
  x_count =  count_vect.transform(df['response']) 
  # Convert to matrix
  x_count = csr_matrix(x_count)
  x_count  = x_count.todense()

  return x_count

def tfidf_features(df):


### word level tf-idf


  tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=3000) # considers the top 5000 most frequent features
  tfidf_vect.fit(df['response'])

  tfidf =  tfidf_vect.transform(df['response'])
  tfidf = csr_matrix(tfidf)

  tfidf  = tfidf.todense()

  return tfidf

def tfidf_ngram_features(df):

  ### ngram level tf-idf

  tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,2), max_features=3000)
  tfidf_vect_ngram.fit(df['response'])

  tfidf_ngram =  tfidf_vect_ngram.transform(df['response'])
  tfidf_ngram = csr_matrix(tfidf_ngram)
  tfidf_ngram  = tfidf_ngram.todense()

  return tfidf_ngram

def tfidf_ngram_chars(df):

  ### characters level tf-idf

  tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(1,2), max_features=1500)
  tfidf_vect_ngram_chars.fit(df['response'])

  tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(df['response']) 

  tfidf_ngram_chars = csr_matrix(tfidf_ngram_chars)
  tfidf_ngram_chars  = tfidf_ngram_chars.todense()

  return tfidf_ngram_chars

def ling_feature_vec(norm_features_df):
  from numpy import array
  from scipy.sparse import csr_matrix
  x = norm_features_df.copy()

  Features = x.drop(['response', 'type', 'subject','title','date', 'id' ],axis=1)

  Features = Features.to_numpy()

  Features = csr_matrix(Features)

  # reconstruct dense matrix
  Features  = Features .todense()
  return Features

def all_count_features(tfidf,tfidf_ngram,tfidf_ngram_chars, x_count):
  all_count_features = np.column_stack([tfidf,tfidf_ngram,tfidf_ngram_chars, x_count])

  return all_count_features

def all_features(x_count,tfidf,tfidf_ngram,tfidf_ngram_chars, Features):
  all_features = np.column_stack([x_count,tfidf,tfidf_ngram,tfidf_ngram_chars, Features])
  return all_features
