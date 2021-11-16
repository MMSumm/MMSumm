import pandas as pd 
from ast import literal_eval
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import spacy
from spacy import displacy
import json
import urllib
nlp = spacy.load('en_core_web_trf')
from tqdm import tqdm
print("lol")
root_dir='/scratch/full_data_test/test_data/'
csv_file='test.csv'
df = pd.read_csv(root_dir+csv_file)
print(len(df['article']))
import os.path
import gensim.downloader as api
wv = api.load('word2vec-google-news-300')

for idx in tqdm(range(len(df['article']))):
    article_file = root_dir+'article/'+df['article'][idx]
    write_file = root_dir+'simple/'+df['article'][idx]
    if not os.path.exists(write_file):
        art = open(article_file)
        art = art.read()
        art = art.replace("\n", " ")
        art = art.rstrip()
        text = art.split("@title ")[1].split("@summary ")[0]
        text = text.split('@body')[1]

        
        doc = nlp(text)
        ents = [ent for ent in doc.ents]
        tokens = [token for token in doc.__iter__()]

        inc = [0 for i in tokens]
        for ent in ents:
            if ent.label_=='DATE' or ent.label_=='GPE':
                ind = ent.start - 1
                if tokens[ind].dep_=='prep':
                    inc[ind]=1
                for i in range(ent.start, ent.end):
                    inc[i]=1

        text1=''
        for i in range(len(tokens)):
            if inc[i]==0:
                text1=text1+tokens[i].text+' '

        doc = nlp(text1)
        ents = [ent for ent in doc.ents]
        tokens = [token for token in doc.__iter__()]

        inc=[0 for i in tokens]
        for ent in ents:
            if ent.label_=='ORG' or ent.label_=='PRODUCT':
                ind = ent.end
                if ind<len(tokens):
                    if tokens[ind].pos_ == 'NOUN':
                        for i in range(ent.start, ent.end):
                            inc[i]=1
            if ent.label_=='PERSON':
                ind = ent.start-1
                if ind>=0:
                    if tokens[ind].pos_ == 'NOUN':
                        for i in range(ent.start, ent.end):
                            inc[i]=1

        text1=''
        for i in range(len(tokens)):
            if inc[i]==0:
                text1=text1+tokens[i].text+' '

        doc = nlp(text1)
        ents = [ent for ent in doc.ents]
        tokens = [token for token in doc.__iter__()]

        f=open('objects_vocab.txt')
        listObj = f.read().split('\n')
        done=[]
        for ent in ents:
            if ent.label_=='PERSON' or ent.label_=='ORG' or ent.label_=='EVENT' or ent.label_=='FAC':
                if tokens[ent.start-1].text=='Mr':
                    text1 = text1.replace(tokens[ent.start-1].text+' '+ent.text, 'man')
                    continue
                if tokens[ent.start-1].text=='Mrs' or tokens[ent.start-1].text=='Ms':
                    text1 = text1.replace(tokens[ent.start-1].text+' '+ent.text, 'woman')
                    continue
                if ent.text not in done:
                    maxSim=0.1
                    rep = ent.text
                    w = ent.text.replace(' ','_')
                    try:
                        vec = wv[w]
                    except KeyError:
                        text1=text1.replace(ent.text, ent.label_.lower())
                        continue
                        
                    for obj in listObj:
                        try:
                            x = wv[obj]
                            sim = wv.similarity(w, obj)
                            if sim > maxSim:
                                maxSum=sim
                                rep=obj
                        except:
                            continue
                    if rep != ent.text:
                        text1=text1.replace(ent.text, rep)
                    else:
                        text1 = text1.replace(ent.text, ent.label_.lower())
                    done.append(ent.text)

        doc = nlp(text1)
        ents = [ent for ent in doc.ents]
        tokens = [token for token in doc.__iter__()]

        inc = [0 for i in tokens]
        for ent in ents:
            if ent.label_=='DATE' or ent.label_=='GPE':
                ind = ent.start - 1
                if tokens[ind].dep_=='prep':
                    inc[ind]=1
                for i in range(ent.start, ent.end):
                    inc[i]=1

        text1=''
        for i in range(len(tokens)):
            if inc[i]==0:
                text1=text1+tokens[i].text+' '

        text1 = text1.replace(', ', ' ')

        writeF = open(write_file, 'w')
        writeF.write(text1)

        
        

