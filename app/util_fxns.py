from time import time
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

from sklearn.metrics.pairwise import euclidean_distances

import regex as re
import requests
from bs4 import BeautifulSoup
import nltk
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('words')
nltk.download('stopwords')

import pprint

from nltk.tag.perceptron import PerceptronTagger

import pickle
from io import BytesIO  
import sys

import json
import base64
import urllib

np.random.seed(42)


from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import FreqDist
from collections import Counter
import pandas as pd

sid = SentimentIntensityAnalyzer()
punc = re.compile('\p')




def get_normalized_and_split_data(all_data,x_names,split=0.2,by_artist=False):

    if by_artist:
        artists = all_data['artist'].unique()
        X_train = all_data[all_data['artist'].isin(artists[:-2])][x_names].values
        y_train = list(all_data[all_data['artist'].isin(artists[:-2])]['labels'])

        X_test = all_data[all_data['artist'].isin(artists[-2:])][x_names].values
        y_test = list(all_data[all_data['artist'].isin(artists[-2:])]['labels'])
    else:
        y = list(all_data['labels'])
        X = all_data[x_names].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=3)
    
    scaler = StandardScaler()
    X_train =scaler.fit_transform(X_train)
    if len(y_test)>0:
        X_test = scaler.transform(X_test)

    to_drop=[]
    for i in range(X_train.shape[0]):
        for j in range(X_train.shape[1]):
            if abs(X_train[i][j])>5:
                to_drop.append(i)

    to_drop=list(set(to_drop))

    X_train = np.delete(X_train, to_drop, axis=0)
    y_train = np.delete(y_train, to_drop, axis=0)

    n_samples, n_features = X_train.shape

    return X_train, X_test, y_train, y_test, scaler
    
def get_euc_dist(set1,set2,set1_y,set2_y,feature_names,n_top=10):
    ed_df = pd.DataFrame()
    ed = euclidean_distances(set1, set2)
    
    for n in range(len(set1_y)):
        df=pd.DataFrame()
        df['distance']=ed[n,:]
        df['to']=set2_y
        df['from']=set1_y[n]
        df['to_ind']=range(len(set2_y))
        df['from_ind']=n
        df_feat = pd.DataFrame(set2, columns=feature_names)
        df = pd.concat([df, df_feat], axis=1)
        ed_df = ed_df.append(df)

    cols = ['from','to','distance'] + feature_names
    ed_df = ed_df[cols]
    ed_df = ed_df.sort_values(['from','distance'],ascending=True)
    
    ed_df_top = ed_df.groupby('from').head(n_top)
    ed_df_top['rel_conf'] = ed_df_top['distance']/ed_df_top['distance'].max()
    ed_df_top['rel_conf'] = ed_df_top['rel_conf']-ed_df_top['rel_conf'].min()
    ed_df_top['rel_conf'] = (1.0-ed_df_top['rel_conf'])*100
    ed_df_top = ed_df_top.sort_values(['distance'],ascending=True)
    ed_df_top['Rank'] = range(1,len(ed_df_top.index)+1)
    ed_df_top['distance'] =ed_df_top['distance'].round(2)
    ed_df_top = ed_df_top.rename(columns={'distance': 'Distance', 'to': 'My Songs', 'rel_conf': 'Relative_Confidence'})
    ed_df_top['My Song'], ed_df_top['Artist'] = ed_df_top['My Songs'].str.split('-', 1).str
    return ed_df_top[["Rank",'Artist','My Song','Distance']], ed_df_top

def ie_preprocess(document):
    
    tagger = PerceptronTagger() 
    tagged = []
    sentences = document.split("\n")
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    for sent in sentences:
        tagged_tokens = tagger.tag(sent)
        tagged.append(tagged_tokens)
    
    return tagged

def tokenize_song(lyrics):
    tokenized_lyrics = ie_preprocess(lyrics)
    return tokenized_lyrics

def unique_words(text):
    return len(set(text))*1.0

def total_words(text):
    return len(text)*1.0

def lexical_diversity(text):
    return unique_words(text) / total_words(text)

def unusual_words(text):
    text_vocab = set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab - english_vocab
    return len(unusual)

def content_fraction(text):
    stopwords = nltk.corpus.stopwords.words('english')
    content = [w for w in text if w.lower() not in stopwords]
    return len(content) / total_words(text)

def avg_len_words(text):
    avg_len =0.0
    for t in text:
        avg_len+=len(t)
    return avg_len/len(text)

def get_pos_stats(tagged):
    stats =[]
    names =[]
    counts = Counter(tag for word,tag in tagged)
    total = sum(counts.values())
    counts_perc = dict((word, float(count)/total) for word,count in counts.items())
    for t in ['NN','PR','RB','RBR','RBS','UH','VB','JJ','JJR','JJS','EX']:
        names.append(t)
        pa = 0.0
        for key in counts_perc.keys():
            if key.startswith(t):
                pa+=counts_perc[key]
        stats.append(pa)

    return stats, names

def get_word_stats(input_text):
    
    text=[]
    for t in input_text:
        if punc.search(t):
            continue
        text.append(t)
    stats = []
    stats.append(unique_words(text))
    stats.append(total_words(text))
    stats.append(lexical_diversity(text))
    stats.append(unusual_words(text))
    stats.append(content_fraction(text))
    stats.append(avg_len_words(text))
    names = ['unique_words', 'total_words','lex_div','unusual_words','content_frac','avg_len_words']
    text_joined = " ".join(text)
    ss = sid.polarity_scores(text_joined)
    for k in sorted(ss):
        stats.append(ss[k])
        names.append(k)
    return stats, names

def get_song_data(tokenized_song):

    lyrics = tokenized_song
    #get sentences without tokens
    lyrics_sep = []
    for y in lyrics:
        tmpsent = [word for (word,tag) in y]
        lyrics_sep = lyrics_sep + tmpsent

    if len(lyrics_sep)==0:
        return [], []

    x_data = []
    x_names = []
    tagged_lyrics = [j for i in lyrics for j in i]
    #get word
    stats, names = get_word_stats(lyrics_sep)
    x_names = x_names + names
    x_data = x_data + stats

    stats, names = get_pos_stats(tagged_lyrics)
    x_names = x_names + names
    x_data = x_data + stats
    return x_data, x_names

def get_custom_text_lyric(text):
    text_stripped = re.sub(' +',' ',text)
    text_stripped = text_stripped.replace("\n\n","\n")
    text_stripped = text_stripped.replace("\r","\n")
    return text_stripped