#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#data preprocessing
#import libraries


import pandas as pd
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from keras.preprocessing import text, sequence
import pandas, xgboost, numpy, textblob, string

from keras import layers, models, optimizers
from sklearn.metrics import f1_score
import nltk

from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.stem import PorterStemmer
from collections import Counter
import string
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import tree

import matplotlib.pyplot as plt
import re
from collections import Counter
import sys
print(sys.executable)

from scipy.sparse import hstack

pd.set_option('display.max_columns',11)
data = open('',encoding='utf8').read()
labels, texts= [], []
for i, line in enumerate(data.split("\n")):  # enumerates entries, splits each tweet into a list item
    content = line.split()
    labels.append(content[0]) # get labels
    texts.append(" ".join(content[1:])) # get text
    

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
  response = []
    
    
      # for each file (row) in the df, read in the file 
  for row_i in df.index:
      filename = df.iloc[row_i]['response']
      file_text = process_file(str(filename))
          #append processed text to list
      response.append(file_text)
    
    #add column to the copied dataframe
  text_df['response'] = response

  context_0 = []
  for row_i in df.index:
      filename = df.iloc[row_i]['context/0']
      file_text = process_file(str(filename))
          #append processed text to list
      context_0.append(file_text)
    
    #add column to the copied dataframe
  text_df['context/0']=context_0
  context_1= []
  for row_i in df.index:
      filename = df.iloc[row_i]['context/1']
      file_text = process_file(str(filename))
          #append processed text to list
      context_1.append(file_text)
    
    #add column to the copied dataframe
  text_df['context/1']= context_1
                                
  concat_tweet = []
  for row_i in df.index:
      filename = df.iloc[row_i]['concat_tweet']
      file_text = process_file(str(filename))
          #append processed text to list
      concat_tweet.append(file_text)
    
    #add column to the copied dataframe
  text_df['concat_tweet']= concat_tweet                           
                                
    
  return text_df





# In[ ]:


#tokenizer
trainDF['tokenized_text'] = trainDF.apply(lambda row: nltk.word_tokenize(row['text']), axis=1)


# In[ ]:


#count features as count vector
## load the dataset
data = open('').read()
labels, texts = [], []
for i, line in enumerate(data.split("\n")): # enumerates entries
    content = line.split()
    labels.append(content[0]) # get labels
    texts.append(" ".join(content[1:])) # get text

# create a dataframe using texts and lables
trainDF = pandas.DataFrame() 
trainDF['text'] = texts
trainDF['label'] = labels

# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y =  model_selection.train_test_split(trainDF['text'],trainDF['label'], train_size=0.6,test_size=0.4)


# label encode the target variable for ML --> numeric
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)


# create a matrix. count_vect: every row represents a document from the corpus;
# every column is a term from the corpus;
# every cell represents the freq count of a particular term in particular document
 
# create a count vectorizer object for the training data
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}') # word that appears 1 or more times
count_vect.fit(trainDF['text'])

# transform the training and validation data using count vectorizer object: doc x term
xtrain_count =  count_vect.transform(train_x) 
xvalid_count =  count_vect.transform(valid_x)


#### Extracting features ########

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000) # considers the top 5000 most frequent features
tfidf_vect.fit(trainDF['text'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)

# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(trainDF['text'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(trainDF['text'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x) 

## Word embeddings 

# https://fasttext.cc/docs/en/english-vectors.html
# load the pre-trained word-embedding vectors 

embeddings_index = {}
for i, line in enumerate(open('wiki-news-300d-1M.vec')):
    values = line.split()
    embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')

# create a tokenizer 
token = text.Tokenizer()
token.fit_on_texts(trainDF['text'])
word_index = token.word_index

# convert text to sequence of tokens and pad them to ensure equal length vectors 
train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)

# create token-embedding mapping
embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        

trainDF['char_count'] = trainDF['text'].apply(len)
trainDF['word_count'] = trainDF['text'].apply(lambda x: len(x.split()))
trainDF['word_density'] = trainDF['char_count'] / (trainDF['word_count']+1)
trainDF['punctuation_count'] = trainDF['text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 
trainDF['title_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
trainDF['upper_case_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))


# In[ ]:


#LIWC


from liwc import Liwc
import csv 

lwc = Liwc("liwc_dictionaries_shared/LIWC2007_English100131.dic")
liwcresults =[]

for token in trainDF['tokenized_text']:
    results = (lwc.parse(token))
    liwcresults.append(results)

liwc_keys = []
for i in range(len(liwcresults)):
    for k in liwcresults[i].keys():
        if k not in liwc_keys:
            liwc_keys.append(k)
print(liwc_keys)
    

#look at each category in range 3,000, if the value is not in the list of keys, append new key. 

with open('.csv', 'w') as csv_file:  
    dict_writer = csv.DictWriter(csv_file, liwc_keys)       
    writer = csv.writer(csv_file)
    dict_writer.writeheader()
    dict_writer.writerows(liwcresults)
    
liwcdata = pd.read_csv(".csv") 
liwcdata.head()
trainDF.head()

trainDF = pd.concat([liwcdata, trainDF], axis=1, sort=False)

trainDF.fillna(0, inplace=True)
#trainDF1 = trainDF1.astype(float)

trainDF.to_csv('liwc.csv')


# In[ ]:


#VADER


def vader(sentence):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()
    score = vader.polarity_scores(sentence)
    return score

# Lits to generate a positive, negative, neutral, and compound score
vs_compound = []
vs_pos = []
vs_neu = []
vs_neg = []


for row in trainDF["text"]:
    score = vader(row)

    neg = float(score['neg'])
    vs_neg.append(neg)
    neu =float(score['neu'])
    vs_neu.append(neu)
    pos =float(score['pos'])
    vs_pos.append(pos)
    compound = float((score['compound']+1)/2) # rescaling to the [0,1] range
    vs_compound.append(compound)
   # vader_df = pd.DataFrame(trainDF["text"])      

trainDF["vader_pos"] = vs_pos
trainDF["vader_neg"] = vs_neg
trainDF["vader_neu"] = vs_neu
trainDF["vader_compound"] = vs_compound


# In[ ]:


#VAD
import csv
import sys
import os
import statistics
import time
import argparse

from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP()

from nltk import tokenize
from nltk.corpus import stopwords

stops = set(stopwords.words("english"))
anew = "../lib/EnglishShortened.csv"


# performs sentiment analysis on inputFile using the ANEW database, outputting results to a new CSV file in outputDir
def analyzefile(input_file, output_dir, mode):
    """
    Performs sentiment analysis on the text file given as input using the ANEW database.
    Outputs results to a new CSV file in output_dir.
    :param input_file: path of .txt file to analyze
    :param output_dir: path of directory to create new output file
    :param mode: determines how sentiment values for a sentence are computed (median or mean)
    :return:
    """
    output_file = os.path.join(output_dir, "Output Anew Sentiment " + os.path.basename(input_file).rstrip('.txt') + ".csv")

    # read file into string
    with open(input_file, 'r') as myfile:
        fulltext = myfile.read()
    # end method if file is empty
    if len(fulltext) < 1:
        print('Empty file.')
        return

    from nltk.stem.wordnet import WordNetLemmatizer
    lmtzr = WordNetLemmatizer()

    # otherwise, split into sentences
    sentences = tokenize.sent_tokenize(fulltext)
    i = 1 # to store sentence index
    # check each word in sentence for sentiment and write to output_file
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['Sentence ID', 'Sentence', 'Sentiment', 'Sentiment Label', 'Arousal', 'Dominance',
                      '# Words Found', 'Found Words', 'All Words']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # analyze each sentence for sentiment
        for s in sentences:
            # print("S" + str(i) +": " + s)
            all_words = []
            found_words = []
            total_words = 0
            v_list = []  # holds valence scores
            a_list = []  # holds arousal scores
            d_list = []  # holds dominance scores

            # search for each valid word's sentiment in ANEW
            words = nlp.pos_tag(s.lower())
            for index, p in enumerate(words):
                # don't process stops or words w/ punctuation
                w = p[0]
                pos = p[1]
                if w in stops or not w.isalpha():
                    continue

                # check for negation in 3 words before current word
                j = index-1
                neg = False
                while j >= 0 and j >= index-3:
                    if words[j][0] == 'not' or words[j][0] == 'no' or words[j][0] == 'n\'t':
                        neg = True
                        break
                    j -= 1

                # lemmatize word based on pos
                if pos[0] == 'N' or pos[0] == 'V':
                    lemma = lmtzr.lemmatize(w, pos=pos[0].lower())
                else:
                    lemma = w

                all_words.append(lemma)

                # search for lemmatized word in ANEW
                with open(anew) as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        if row['Word'].casefold() == lemma.casefold():
                            if neg:
                                found_words.append("neg-"+lemma)
                            else:
                                found_words.append(lemma)
                            v = float(row['valence'])
                            a = float(row['arousal'])
                            d = float(row['dominance'])

                            if neg:
                                # reverse polarity for this word
                                v = 5 - (v - 5)
                                a = 5 - (a - 5)
                                d = 5 - (d - 5)

                            v_list.append(v)
                            a_list.append(a)
                            d_list.append(d)

            if len(found_words) == 0:  # no words found in ANEW for this sentence
                writer.writerow({'Sentence ID': i,
                                 'Sentence': s,
                                 'Sentiment': 'N/A',
                                 'Sentiment Label': 'N/A',
                                 'Arousal': 'N/A',
                                 'Dominance': 'N/A',
                                 '# Words Found': 0,
                                 'Found Words': 'N/A',
                                 'All Words': all_words
                                 })
                i += 1
            else:  # output sentiment info for this sentence

                # get values
                if mode == 'median':
                    sentiment = statistics.median(v_list)
                    arousal = statistics.median(a_list)
                    dominance = statistics.median(d_list)
                else:
                    sentiment = statistics.mean(v_list)
                    arousal = statistics.mean(a_list)
                    dominance = statistics.mean(d_list)

                # set sentiment label
                label = 'neutral'
                if sentiment > 6:
                    label = 'positive'
                elif sentiment < 4:
                    label = 'negative'

                writer.writerow({'Sentence ID': i,
                                 'Sentence': s,
                                 'Sentiment': sentiment,
                                 'Sentiment Label': label,
                                 'Arousal': arousal,
                                 'Dominance': dominance,
                                 '# Words Found': ("%d out of %d" % (len(found_words), len(all_words))),
                                 'Found Words': found_words,
                                 'All Words': all_words
                                 })
                i += 1


def main(input_file, input_dir, output_dir, mode):
    """
    Runs analyzefile on the appropriate files, provided that the input paths are valid.
    :param input_file:
    :param input_dir:
    :param output_dir:
    :param mode:
    :return:
    """

    if len(output_dir) < 0 or not os.path.exists(output_dir):  # empty output
        print('No output directory specified, or path does not exist')
        sys.exit(0)
    elif len(input_file) == 0 and len(input_dir)  == 0:  # empty input
        print('No input specified. Please give either a single file or a directory of files to analyze.')
        sys.exit(1)
    elif len(input_file) > 0:  # handle single file
        if os.path.exists(input_file):
            analyzefile(input_file, output_dir, mode)
        else:
            print('Input file "' + input_file + '" is invalid.')
            sys.exit(0)
    elif len(input_dir) > 0:  # handle directory
        if os.path.isdir(input_dir):
            directory = os.fsencode(input_dir)
            for file in os.listdir(directory):
                filename = os.path.join(input_dir, os.fsdecode(file))
                if filename.endswith(".txt"):
                    start_time = time.time()
                    print("Starting sentiment analysis of " + filename + "...")
                    analyzefile(filename, output_dir, mode)
                    print("Finished analyzing " + filename + " in " + str((time.time() - start_time)) + " seconds")
        else:
            print('Input directory "' + input_dir + '" is invalid.')
            sys.exit(0)


if __name__ == '__main__':
    # get arguments from command line
    parser = argparse.ArgumentParser(description='Sentiment analysis with ANEW.')
    parser.add_argument('--file', type=str, dest='input_file', default='',
                        help='a string to hold the path of one file to process')
    parser.add_argument('--dir', type=str, dest='input_dir', default='',
                        help='a string to hold the path of a directory of files to process')
    parser.add_argument('--out', type=str, dest='output_dir', default='',
                        help='a string to hold the path of the output directory')
    parser.add_argument('--mode', type=str, dest='mode', default='mean',
                        help='mode with which to calculate sentiment in the sentence: mean or median')
    args = parser.parse_args()

    # run main
    sys.exit(main(args.input_file, args.input_dir, args.output_dir, args.mode))


# In[ ]:


#Emotional embeddings


## Function to get Emotion Vectors
from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion")

model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-emotion")

def get_emotion(text):
  input_ids = tokenizer.encode(text + '</s>', return_tensors='pt')

  output = model.generate(input_ids=input_ids,
               max_length=2)

  dec = [tokenizer.decode(ids) for ids in output]
  label = dec[0]
  return input_ids



#Function to get Emotion labels

from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion")

model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-emotion")

def get_emotion_label(text):
  input_ids = tokenizer.encode(text + '</s>', return_tensors='pt')

  output = model.generate(input_ids=input_ids,
               max_length=2)

  dec = [tokenizer.decode(ids) for ids in output]
  label = dec[0]
  return label



##Create feature vector 
new_df["Response_emotion"] = response_sent


new_df.head()


context_sent = []


for row in new_df["context/0"]:
    if row == 0:
        
        sentiment = 'NaN'
  
        context_sent.append(sentiment)
    else:
        
        sentiment = get_emotion_label(row)
        context_sent.append(sentiment)
        


##Create feature vector 
new_df["context0_sentiment"] = context_sent
new_df.to_csv('final_test_emotions.csv')



# In[ ]:


#cosine similarity


from sklearn.metrics.pairwise import cosine_similarity
def get_cosine_similarity(feature_vec_1,feature_vec_2):
    return cosine_similarity(feature_vec_1.reshape(1, -1), feature_vec_2.reshape(1, -1))[0][0]

def get_vectors(text1, text2):
    # Turns text into emotional tensors
    tensor1 = get_emotion(text1)
    tensor2 = get_emotion(text2)
    
    # Turns emotional tensors into numpy array
    Vector_1 = numpy.array(tensor1)
    Vector_2 = numpy.array(tensor2)
    
    #pads lower array with zeros to match higher array
    if Vector_1.size >= Vector_2.size:
        vA = Vector_1
        vB = Vector_2
        result = np.zeros(vA.shape)
        result[:vB.shape[0],:vB.shape[1]] = vB
        vC = vA
    elif Vector_1.size <= Vector_2.size:
        vB = Vector_2
        vA = Vector_1
        result = np.zeros(vB.shape)
        result[:vA.shape[0],:vA.shape[1]]= vA
        vC = vB
    return vC, result

a,b = get_vectors(text2,text1)
cosine_similarities = get_cosine_similarity(a,b)
print(cosine_similarities)


Emotional_sim = []
for row_a in new_df['response']:
    if row_a == 0:
        row_a = 'NaN'
    else:
        row_a = row_a
            
    for row_b in new_df['context/0']:
        if row_b == 0:
            row_b = 'NaN'
        else:
            row_b = row_b
        
        
        text1 = row_a
        text2 = row_b
        
    
        a,b = get_vectors(text1,text2)
        cosine_similarities = get_cosine_similarity(a,b)
        Emotional_sim.append(cosine_similarities)
    
new_df["Emotional_sim_response/context/0"] = Emotional_sim



# In[ ]:


def create_text_column(df):

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


# In[ ]:


#SHAPLEY score

shap.initjs()
xgb_model = XGBRegressor(n_estimators=1000, max_depth=10, learning_rate=0.001, random_state=0)
xgb_model.fit(X_train, y_train)
y_predict = xgb_model.predict(X_test)
mean_squared_error(y_test, y_predict)**(0.5)

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_train)
i = 5000
shap.force_plot(explainer.expected_value, shap_values[i], features=X_train.loc[5000], feature_names=X_train.columns)

shap.summary_plot(shap_values, features=X_train, feature_names=X_train.columns)



# In[ ]:


#classifier #random forest #different experiments

accuracy, precision, recall,fl  = train_model(RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0), xtrain_count, y_train, xvalid_count)
print ("random Forest: ", "A:", round(accuracy,2), "P:", round(precision,2), "R:", round(recall,2), "F1:", round(f1,2))

