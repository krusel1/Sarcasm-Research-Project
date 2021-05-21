# Sarcasm-Research-Project
Title: Automatic Sarcasm Detection Across Two Social Media Platforms
Introduction: This repository explores a computational approach, specially using NLP features, to automatic sarcasm detection across two social media platforms; Twitter and Reddit. The linguistic features that are investigated are a combination of count, stylistic, and psychological features. The features are first run on a corpus of tweets from Twitter and separately on a corpus of Reddit threads, with and without context. The two results of each social media platform are compared. 
Hypotheses: Sarcasm tweets and threads tend to be associated with a negative emotion. The presence of hash tags is crucial for sarcasm detection. Sarcastic responses are typically incongruent in terms of senitment and emotion with their contextual counterpart. Context is absolutely critical information for sarcasm detection. 
Code language: Python 3.6
Python libraries used: LIWC, Vader, NLTK, VAD, textblob, Hugging face (as a pretrained model), SHAPLEY
Critical findings: This research proves the hypothesis that contextual information is absolutely necessary for sarcasm detection. When response was only considered, the classifer performed much lower. The more hashtags in the data set, the higher the results of the classifier. Hash tags appear to be an indicator of sarcasm. Lastly, sarcastic tweets and threads are associated with a negative emotion; anger, as well as their contextual counterpart. Non-sarcastic contexts typically reply with a positive emotion;joy. 

Will need:
Download pre-trained word vectors...https://fasttext.cc/docs/en/english-vectors.html
Hugging face pretrained model...https://huggingface.co/mrm8488/t5-base-finetuned-emotion
