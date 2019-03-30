from time import time
import numpy as np

from sklearn.cluster import KMeans
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

from util_fxns import get_song_data, ie_preprocess, get_euc_dist, tokenize_song



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

def tokenize_corpus(lyric_corpus_file, tokenized_corpus_file="tokenized_lyric_corpus.p"):
    try:
        tokenized_lyric_corpus = pickle.load(open(tokenized_corpus_file, 'rb'))
    except:
        tokenized_lyric_corpus = {}

    lyric_corpus = pickle.load(open(lyric_corpus_file, 'rb'))
    for a in lyric_corpus.keys():
        if not (a in tokenized_lyric_corpus):
            tokenized_lyric_corpus[a] = {}
        print("ARTIST:", a)
        for s in lyric_corpus[a].keys():
            if s in tokenized_lyric_corpus[a]:
                continue
            lyrics = lyric_corpus[a][s]
            print("---SONG:",s)
            tokenized_lyrics = ie_preprocess(lyrics)
            tokenized_lyric_corpus[a][s] = tokenized_lyrics
            pickle.dump(tokenized_lyric_corpus, open(tokenized_corpus_file, 'wb'))
    
    return tokenized_lyric_corpus

def search_musix_corpus(artists, pages=2,corpus_file='lyric_corpus.p'):
    p = re.compile('\/lyrics\/*')

    webpage = "https://www.musixmatch.com"
    url = 'https://www.musixmatch.com/search/'
    useragent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.75 Safari/537.36'

    try:
        lyric_corpus = pickle.load(open(corpus_file, 'rb'))
    except:
        lyric_corpus = {}

    for a in artists:
        artist_lyrics = {}
        searchlink = url + a +"/artists"
        r  = requests.get(searchlink, headers={'User-Agent': useragent})
        soup = BeautifulSoup(r.content, "lxml")
        gdata = soup.find_all('a',{'class':'cover'})
        try:
            artist_page = webpage + str(gdata[0].get('href')) 
            a = gdata[0].text
        except:
            print(searchlink)
            print(r)
            print(gdata)
            continue


        for i in range(1,pages+1):
            artist_page_num = artist_page
            if i>1:
                artist_page_num = artist_page+"/"+str(i)
            print(artist_page_num)

            try:
                r  = requests.get(artist_page_num, headers={'User-Agent': useragent})
            except:
                print("No lyrics page {i}, leaving this artist.".format(i=i))
                break
            soup = BeautifulSoup(r.content, "lxml")
            gdata = soup.find_all('a',{'class':'title'})

            for s in gdata:
                slink = str(s.get('href'))
                if p.search(slink):
                    song_lyrics = []
                    song_page = webpage+slink
                    song_name = s.span.text + "-" + a

    #                 print(song_page)
                    try:
                        r  = requests.get(song_page, headers={'User-Agent': useragent})
                    
                        soup = BeautifulSoup(r.content, "lxml")
                        lyrics = soup.find_all('p',{'class':'mxm-lyrics__content'})
      
                        for l in lyrics:
                            song_lyrics.append(l.text)

                        song_lyrics  = "\n".join(song_lyrics)
                        song_lyrics = song_lyrics.replace("\n\n","\n")
                        artist_lyrics[song_name]=song_lyrics
                    except:
                        continue
    
        lyric_corpus[a] = artist_lyrics

    pickle.dump(lyric_corpus, open(corpus_file, 'wb'))
    return lyric_corpus

def get_corpus_dataframe(tokenized_lyric_file, output_file="dataframe_storage.csv"):
    tokenized_lyric_corpus = pickle.load(open(tokenized_lyric_file, 'rb'))
    X_data = []
    y_labels=[]
    y_artists=[]
    x_final_names = []
    for a in tokenized_lyric_corpus.keys():
        total_songs = len(tokenized_lyric_corpus[a].keys())
        curr_song=0
        for s in tokenized_lyric_corpus[a].keys():
            x_data, x_names = get_song_data(tokenized_lyric_corpus[a][s])
            if len(x_data)==0:
                continue
            x_final_names =x_names
            X_data.append(x_data)
            y_labels.append(s)
            y_artists.append(a)

    all_data = pd.DataFrame(X_data)
    all_data.columns=x_final_names
    all_data['labels']=y_labels
    all_data['artist']=y_artists
    all_data.to_csv(output_file,  encoding='utf-8')

    return all_data


if __name__ == '__main__':
    data_only=False
    if len(sys.argv)>1 and sys.argv[1]=='dataonly':
        data_only = True

    suffix = sys.argv[2]
    pgs = int(sys.argv[3])
    storagepath = "static/"
    tlc =storagepath+"tokenized_lyric_{suffix}.p".format(suffix=suffix)
    lc=storagepath+'lyric_{suffix}.p'.format(suffix=suffix)
    ds = storagepath+"df_{suffix}.csv".format(suffix=suffix)
    artists = ["julia jacklin", "mitski", "margaret glaspy", "big thief", "city calm down",
    "alvvays", "EL VY", "Heartless Bastards", "Julien Baker", "St. Vincent", "real estate",
    "sufjan stevens", "conor oberst", "monsters of folk", "William Fitzsimmons", "iron & wine",
    "daughter", "lord huron", "andrew bird", "volcano choir", "the head and the heart", "bon iver",
    "sufjan stevens", "london grammar", "wolf alice", "HAIM", "regina spektor", 
    "oh wonder", "the japanese house", "broods", "seafret"]

    artists = set(artists)
    if not data_only:
        lyric_corpus = search_musix_corpus(artists,pages=pgs,corpus_file=lc)
        tokenized_lyric_corpus = tokenize_corpus(lc, tokenized_corpus_file=tlc)
        all_data = get_corpus_dataframe(tlc, output_file=ds)
    else:
        ds = "df_{suffix}.csv".format(suffix=suffix)
        all_data = pd.read_csv(ds, encoding="utf-8")

        test_lyric = "Test text to get the x_names field lol what ever"
        #print test_lyric
        tokenized_song = tokenize_song(test_lyric)
        #print tokenized_song
        dummy_var, x_names = get_song_data(tokenized_song)

        X_train, X_test, y_train, y_test, scaler= get_normalized_and_split_data(all_data, x_names,split=0.3)
        reco_df, full_reco_df = get_euc_dist(X_test,X_train,y_test,y_train,x_names,n_top=5)
        #print full_reco_df.head()
