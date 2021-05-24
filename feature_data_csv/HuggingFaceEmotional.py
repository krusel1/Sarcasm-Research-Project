#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 19:41:33 2020

@author: laurenkruse
"""

#hugging face pretrained model
#only creates labels not cosine similarity on emotional embeddings 

import pandas as pd


csv_file = 'Reddit_Training2Contexts.csv'   ## Import data from directory 
df = pd.read_csv(csv_file)

reddit_df=df.copy()

reddit_df.groupby([column_name_lst])["label"]

from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion",use_fast=False)

model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-emotion")

#creating words into tensors 
def get_emotion(text):
  input_ids = tokenizer.encode(text + '</s>', return_tensors='pt')

  output = model.generate(input_ids=input_ids,
               max_length=2)

  dec = [tokenizer.decode(ids) for ids in output]
  label = dec[0]
  return input_ids


df = pd.DataFrame(columns = ['Response Emotion', 'Context/0 Emotion', 'Context/1 Emotion']) 
print(df) 


#putting tensors into labels
def get_emotion_label(text):
  input_ids = tokenizer.encode(text + '</s>', return_tensors='pt')

  output = model.generate(input_ids=input_ids,
               max_length=2)

  dec = [tokenizer.decode(ids) for ids in output]
  label = dec[0]
  return label


response_sent = []
for row in reddit_df["response"]:
    sentiment = get_emotion_label(row)
    
    response_sent.append(sentiment)

df['Response Emotion']=response_sent


#creating feature vector
reddit_df["reponse_emoton"] = response_sent



context0_sent = []
for row in reddit_df["context/0"]:
    sentiment = get_emotion_label(row)
    
    response_sent.append(sentiment)
    
reddit_df["context/0_emotion"] = context0_sent


context1_sent = []

for row in reddit_df["context/1"]:
    sentiment = get_emotion_label(row)
    
    response_sent.append(sentiment)

#creating feature vector
reddit_df["context/1_emotion"] = context1_sent

print(reddit_df)

reddit_df.to_csv('Emotions.csv')







  
 