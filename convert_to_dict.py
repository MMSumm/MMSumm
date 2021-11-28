# Convert data to json dictionaries
# Provide path to MSMO train data

import os
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import json

root_dir='/scratch/full_data_test/test_data/'

from tqdm import tqdm

art_dir = os.listdir(root_dir+'article/')
img_dir = os.listdir(root_dir+'img/')

for idx in tqdm(range(len(art_dir))):
    fname = art_dir[idx]
    article_file = root_dir+'article/'+fname
    art = open(article_file)
    art = art.read()
    art = art.replace("\n", " ")
    art = art.rstrip()
    text = art.split("@title ")[1].split("@summary ")[0]
    text = text.split('@body')[1]
    sents = sent_tokenize(text)
    src = [word_tokenize(sent) for sent in sents]

    summaries = art.split("@title ")[1].split("@summary ")[1:]
    summary = ""
    for i in summaries:
        i = i.rstrip()
        summary+=i
        summary+=". "
    
    sents = sent_tokenize(summary)
    trg = [word_tokenize(sent) for sent in sents]

    simple_file = root_dir+'simple/'+fname
    art = open(simple_file)
    art = art.read()
    sents = sent_tokenize(art)
    smp = [word_tokenize(sent) for sent in sents]

    imgs = []
    for i in img_dir:
        if fname[:-4] in i:
            imgs.append(i)
    k=len(imgs)

    # Filter out images if > 7
    for i in range(7-k):
        imgs.append('')
    data_pt = {'src':src, 'smp':smp, 'trg':trg, 'imgs':imgs}
    fname = "/scratch/anshul/train."+str(idx)+".json"
    f = open(fname, 'w')
    lol = json.dumps(data_pt)
    f.write(lol)
    f.close()
