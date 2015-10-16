__author__ = 'kshitij'

import twitter
import json
import pandas as pd
import numpy as np
import csv
import re
import scipy
import math
import matplotlib
#matplotlib.use("qt4agg")
from matplotlib import pylab
from pylab import *
from matplotlib import pyplot as plt
import os
os.environ['NLTK_DATA']='C:\Users\Indraneel\AppData\Roaming'
from nltk import stem
from nltk.corpus import stopwords
from collections import Counter
from nltk.corpus import stopwords
import string
from nltk import bigrams
from collections import defaultdict


def processTweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    tweet = re.sub('[\s]+', ' ', tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    tweet = tweet.strip('\'"')
    return tweet
processTweet(" !!! @LordSnow: If Jon Snow dies, we riot ")

def replaceTwoOrMore(s):

    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)

stopWords=[]

def getStopWordList(stopWordListFileName):
    stopWords = []
    stopWords.append('AT_USER')
    stopWords.append('URL')

    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords
stemmer=stem.SnowballStemmer("english")


from nltk.tokenize import word_tokenize
import re


emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]

tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)

def tokenize(s):
    return tokens_re.findall(s)

def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

'''
def getFeatureVector(tweet):
    featureVector = []
    words = tweet.split()
    for w in words:
        w = stemmer.stem(w)
        w = replaceTwoOrMore(w)
        w = w.strip('\'"?,.')
        w= w.decode("utf-8")
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)


        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return featureVector
#print getFeatureVector(" !!! @LordSnow: If Jon Snow dies, we riot ")
'''
def sentiment(text):

    words = pattern_split.split(text.lower())
    sentiments = map(lambda word: afinn.get(word, 0), words)
    if sentiments:

        sentiment = float(sum(sentiments))/math.sqrt(len(sentiments))

    else:
        sentiment = 0
    return sentiment


A=[]
F=[]
J=[]
Sa=[]
Su=[]
D=[]

X1=[]
X2=[]
X3=[]
X4=[]
X5=[]
X6=[]


angerwords=[]
filename='emo_anger_s.txt'
fp=open(filename,'r')
line=fp.readline()
while line:
    word=line.strip()
    angerwords.append(word)
    line=fp.readline()
    #print angerwords
fp.close()

#print sentiment(" !!! @LordSnow: If Jon Snow dies, we riot ")

def anger(tweet) :

    X=[]

    for w in tweet :

        if w in angerwords:
            X.append(w)
            X1.append(w)
    A.append(len(X))


joywords=[]
filename='emo_joy_s.txt'
fp=open(filename,'r')
line=fp.readline()
while line:
    word=line.strip()
    joywords.append(word)
    line=fp.readline()
fp.close()

#print anger(" !!! @LordSnow: If Jon Snow dies, we riot ")

def joy(tweet) :

    X=[]

    for w in tweet :
        if w in joywords:
            X.append(w)
            X2.append(w)
    J.append(len(X))


sadwords=[]
filename='emo_sadness_s.txt'
fp=open(filename,'r')
line=fp.readline()
while line:
    word=line.strip()
    sadwords.append(word)
    line=fp.readline()
fp.close()
#print joy(" !!! @LordSnow: If Jon Snow dies, we riot ")



def sadness(tweet) :

    X=[]

    for w in tweet :
        if w in sadwords:
            X.append(w)
            X3.append(w)
    Sa.append(len(X))


fearwords=[]
filename='emo_fear_s.txt'
fp=open(filename,'r')
line=fp.readline()
while line:
    word=line.strip()
    fearwords.append(word)
    line=fp.readline()
fp.close()


def fear(tweet) :

    X=[]

    for w in tweet :
        if w in fearwords:
            X.append(w)
            X4.append(w)
    F.append(len(X))


diswords=[]
filename='emo_disgust_s.txt'
fp=open(filename,'r')
line=fp.readline()
while line:
    word=line.strip()
    diswords.append(word)
    line=fp.readline()
fp.close()


def disgust(tweet) :

    X=[]

    for w in tweet :
        if w in diswords:
            X.append(w)
            X5.append(w)
    D.append(len(X))


surwords=[]
filename='emo_surprise_s.txt'
fp=open(filename,'r')
line=fp.readline()
while line:
    word=line.strip()
    surwords.append(word)
    line=fp.readline()
fp.close()


def surprise(tweet) :

    X=[]

    for w in tweet :
        if w in surwords:
            X.append(w)
            X6.append(w)
    Su.append(len(X))


#fp = open('dis.txt', 'r')
#line = fp.readline()


filenameAFINN = 'AFINN-111.txt'
afinn = dict(map(lambda (w, s): (w, int(s)), [
            ws.strip().split('\t') for ws in open(filenameAFINN) ]))


pattern_split = re.compile(r"\W+")

def main():




    text_file = open("Output.txt", "w")
    #text_file.write(sta)
    S=[]
    text_file.close()
    import text_utils as tu
    import pandas as pd
    df = pd.read_csv('/home/kshitij/Downloads/Valar Morghulis.csv')
    saved_column = df['CONTENT']
    print len(saved_column)
    features=[]
    for i in range(0,len(saved_column)):

            line= df.ix[i,1]
            processedTweet = processTweet(line)
            #print processedTweet

            preprocessed = preprocess(processedTweet)
            #print
            punctuation = list(string.punctuation)
            stop = stopwords.words('english') + punctuation + ['rt', 'via','AT_USER','know','jon','snow','URL']
            featureVector = [term for term in preprocessed
              if term not in stop and
              not term.startswith(('#', '@'))]
            features=featureVector+features
            #print terms_only

            anger(featureVector)
            joy(featureVector)
            fear(featureVector)
            sadness(featureVector)
            disgust(featureVector)
            surprise(featureVector)
            sentiments=sentiment(line)
            S.append(sentiments)
            wf = tu.word_freq(featureVector)
            #print Counter(featureVector).most_common(2)

            #line = fp.readline()
    #print len(features)
    #print type(X1)

    f = open("anger ad.txt", "w")

    f.write("\n".join(map(lambda x: str(x), X1)))
    f.close()

    f = open("surprise ad.txt", "w")

    f.write("\n".join(map(lambda x: str(x), X6)))
    f.close()

    f = open("joy ad.txt ", "w")

    f.write("\n".join(map(lambda x: str(x), X2)))
    f.close()

    f = open("sad ad.txt ", "w")

    f.write("\n".join(map(lambda x: str(x), X3)))
    f.close()

    f = open("fear ad.txt ", "w")

    f.write("\n".join(map(lambda x: str(x), X4)))
    f.close()

    f = open("disgust ad.txt ", "w")

    f.write("\n".join(map(lambda x: str(x), X5)))
    f.close()


    #wf = tu.word_freq(features)
    #print wf
    #sorted_wf = tu.sort_freqs(wf)
    #print sorted_wf


    j=pd.DataFrame(J)
    sa=pd.DataFrame(Sa)
    a=pd.DataFrame(A)
    su=pd.DataFrame(Su)
    d=pd.DataFrame(D)
    f=pd.DataFrame(F)

    data=pd.concat([j,sa,a,su,d,f],axis=1)
    data.columns=['joy','sadness','anger','suprise','disgust','fear']

    jj=0
    count=data.anger.size
    i=0
    while i < count:
            if data['joy'][i]!=0:
                jj=jj+data['joy'][i]
                i=i+1
            else:
                i=i+1
    joy_percent=100.0*jj/count
    k=0
    ss=0


    while k < count:
            if data['sadness'][k]!=0:
                ss=ss+data['sadness'][k]
                k=k+1
            else:
                k=k+1
    sadness_percent=100.0*ss/count


    l=0
    aa=0
    while l < count:
            if data['anger'][l]!=0:
                aa=aa+data['anger'][l]
                l=l+1
            else:
              l=l+1

    anger_percent=100.0*aa/count

    m=0
    uu=0
    while m < count:
        if data['suprise'][m]!=0:
            uu=uu+data['suprise'][m]
            m=m+1
        else:
            m=m+1
    surprise_percent=100.0*uu/count

    n=0
    dd=0
    while n < count:
        if data['disgust'][n]!=0:
            dd=dd+data['disgust'][n]
            n=n+1
        else:
            n=n+1
    disgust_percent=100.0*dd/count


    o=0
    ff=0
    while o < count:
        if data['fear'][o]!=0:
            ff=ff+data['fear'][o]
            o=o+1
        else:
            o=o+1
    fear_percent=100.0*ff/count
    print "The no words of tweets that are joyful is",len(X1)
    print "The percentage of tweets that are sad is",len(X2)
    print "The percentage of tweets that are angry is",len(X3)
    print "The percentage of tweets that show surprise is",len(X4)
    print "The percentage of tweets that show disgust is",len(X5)
    print "The percentage of tweets that are fearful is",len(X6)


    s=0
    p=0
    n=0
    nu=0
    while s < count:
        if S[s] > 0:
            p=p+1
            s=s+1
        elif S[s] < 0:
            n=n+1
            s=s+1
        else:
            nu=nu+1
            s=s+1


    p_per=100.0*p/count
    n_per=100.0*n/count
    nu_per=100.0*nu/count
    print "Percentage of tweets having positive sentiment is",p_per
    print "Percentage of tweets having negative sentiment is",n_per
    print "Percentage of tweets having neutral sentiment is",nu_per


    #print S
    #print min(S)
    z=0
    SS=[]
    R=[]
    r=[]
    T=[]
    while z < count :
        if S[z]!=0:
            T.append(S[z])
            SS.append(S[z]-min(S))
            z=z+1
        else:

            z=z+1

    count1=len(SS)
    sd=np.std(T)

    zzz=0
    R0=[]
    R1=[]
    R2=[]
    R3=[]
    R4=[]
    R5=[]
    while zzz < count1:
        if T[zzz] < (-2*sd):
            R0.append(T[zzz])
            zzz=zzz+1
        elif (-2*sd) <= T[zzz] < (-sd):
            R1.append(T[zzz])
            zzz=zzz+1
        elif (-sd) <= T[zzz] < 0:
            R2.append(T[zzz])
            zzz=zzz+1
        elif 0 <= T[zzz] < sd:
            R3.append(T[zzz])
            zzz=zzz+1
        elif sd <= T[zzz] < 2*sd:
            R4.append(T[zzz])
            zzz=zzz+1
        else:
            R5.append(T[zzz])
            zzz=zzz+1
    r0=len(R0)

    r1=len(R1)
    r2=len(R2)
    r3=len(R3)
    r4=len(R4)
    r5=len(R5)
    avg=(r1 + 2.0*r2 + 3.0*r3 + 4.0*r4 + 5.0*r5)/count1
    print "The average rating of people's sentiment out of five is", avg

    fig=plt.figure()
    '''
    x = scipy.arange(6)
    y = scipy.array([joy_percent,sadness_percent,anger_percent,surprise_percent,disgust_percent,fear_percent])

    ax=fig.add_subplot(223)

    ax.bar(x, y, align='center')
    ax.set_xticks(x)
    ax.set_xticklabels(['Joy', 'Sadness', 'Anger', 'Surprise','Disgust','Fear'])
'''


    ax1=fig.add_subplot(222)




    labels = 'Neutral', 'Positive', 'Negative'
    fracs = [nu_per,p_per,n_per]
    explode=(0.05, 0.05, 0.05)

    ax1.pie(fracs, explode=explode, labels=labels,
                    autopct='%1.1f%%', shadow=True, startangle=90)

    x = scipy.arange(6)
    y = scipy.array([r0,r1,r2,r3,r4,r5])




    ax2=fig.add_subplot(221)

    ax2.bar(x, y, align='center')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['0', '1', '2', '3','4','5'])
    plt.show()


if __name__ == '__main__':
    main()
