from nltk.corpus import stopwords
import re
import nltk
from tqdm import tqdm
from nltk.stem import SnowballStemmer
import pandas as pd
import numpy as np
import spacy
nlp = spacy.load("en_core_web_sm")
import string
from sets import *

def remove_dup(df):
    index = df[df.duplicated() == True].index
    df.drop(index, axis = 0, inplace = True)
    index = df[df['text'].duplicated() == True].index
    df.drop(index, axis = 0, inplace = True)
    df.reset_index(inplace=True, drop = True)
    print("shape after removing duplicates : ",df.shape)
    return df

def lemmatization(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

def remove_stop_words(text):

    Text=[i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def Removing_numbers(text):
    text=''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):
    
    text = text.split()

    text=[y.lower() for y in text]
    
    return " " .join(text)

def Removing_punctuations(text):
    ## Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )
    
    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text =  " ".join(text.split())
    return text.strip()

def Removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def expand_contractions(text, contractions_dict=contractions_dict):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Compile regex pattern for contractions
    contractions_re = re.compile(r'\b(%s)\b' % '|'.join(contractions_dict.keys()))
    
    # Replace contractions
    def replace(match):
        return contractions_dict[match.group(0)]
    
    return contractions_re.sub(replace, text)

def preprocess_text(df):
    tqdm.pandas()
    df.text=df.text.progress_apply(lambda text : lower_case(text))
    df.text=df.text.progress_apply(lambda text : expand_contractions(text))
    df.text=df.text.progress_apply(lambda text : remove_stop_words(text))
    df.text=df.text.progress_apply(lambda text : Removing_numbers(text))
    df.text=df.text.progress_apply(lambda text : Removing_punctuations(text))
    df.text=df.text.progress_apply(lambda text : Removing_urls(text))
    df.text=df.text.progress_apply(lambda text : lemmatization(text))
    return df

def process_sentence(sentence):
    sentence= lower_case(sentence)
    sentence= expand_contractions(sentence)
    sentence= remove_stop_words(sentence)
    sentence= Removing_numbers(sentence)
    sentence= Removing_punctuations(sentence)
    sentence= Removing_urls(sentence)
    sentence= lemmatization(sentence)
    return sentence

