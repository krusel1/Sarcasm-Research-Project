# Sarcasm-Research-Project
Title: Automatic Sarcasm Detection Across Two Social Media Platforms




Introduction: This repository explores a computational approach, specially using NLP features, to automatic sarcasm detection across two social media platforms; Twitter and Reddit. The linguistic features that are investigated are a combination of count, stylistic, and psychological features. The features are first run on a corpus of tweets from Twitter and separately on a corpus of Reddit threads, with and without context. The two results of each social media platform are compared. 


Hypotheses: Sarcasm tweets and threads tend to be associated with a negative emotion. The presence of hash tags is crucial for sarcasm detection. Sarcastic responses are typically incongruent in terms of senitment and emotion with their contextual counterpart. Context is absolutely critical information for sarcasm detection. A lower lexical diversity in a coporus produces higher accuracy rates in sarcasm detection. 

Will need:

Download pre-trained word vectors...https://fasttext.cc/docs/en/english-vectors.html

Hugging face pretrained model...https://huggingface.co/mrm8488/t5-base-finetuned-emotion

Data collected from: https://github.com/EducationalTestingService/sarcasm

Code language: Python 3.6

Python libraries used: LIWC, Vader, NLTK, textblob, SHAPLEY

*Both test sets are made available on this page, including the label as well as only contxt /0 - context /1. Training data can be found as well as the full testing data on the github of the shared task. https://github.com/EducationalTestingService/sarcasm

Critical findings: This research proves the hypothesis that contextual information is absolutely necessary for sarcasm detection. When response was only considered, the classifer performed much lower. The more hashtags in the data set, the higher the results of the classifier. Hash tags appear to be an indicator of sarcasm. Lastly, sarcastic tweets and threads are associated with a negative emotion; anger, as well as their contextual counterpart. Non-sarcastic contexts typically reply with a positive emotion;joy. 


Description of all experiments run
## Notebooks:
Column Name | Description
------------|-------------
`Baseline Classification Results Context + Response.ipynb`|Final notebook for Linguistic feature and Count features for response and context
`Baseline Classification Results CountVectors.ipynb`|Final notebook for Linguistic feature and Count features for response and context
`Baseline Classification Results Response.ipynb`|Final notebook for Linguistic feature and Count features for response
`response + context Classification results ALL features.ipynb`|Final notebook for Linguistic feature and Count features for response and context
`response + context Classification results Select Features.ipynb`|Final notebook for select Linguistic features and Count features for response
`response + context Classification results ALL features.ipynb`|Final notebook for Linguistic feature and Count features for response and context
`Response only Classification results Select Features.ipynb`|Final notebook for select Linguistic features and Count features for response


*Original paper published in 2020 based of the Twitter experiments is listed as a file under  'Final_Sarcasm_CLiC_IT'
*Paper on the current work is located under the file "Research_for_Comp__Ling (1).pdf" 
