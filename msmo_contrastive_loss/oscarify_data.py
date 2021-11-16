import json
import os
import torch
from tqdm import tqdm
import sys


os.chdir('/scratch/anshul')

files = os.listdir()
files.sort()

all_data = []

FILE_NUM = int(sys.argv[1])

from get_sentence_scores import OscarScorer
scorer = OscarScorer(torch.device('cuda'))

dataset = []

cnt = 0

partition = int(len(files)/4) + 1
start = partition*FILE_NUM
end = start + partition

#files = files[start:end]

for file in tqdm(files):
    datapoint = json.load(open(file, 'r'))
    sents = []
    for sent in datapoint['smp']:
        sents.append(' '.join(sent))
    
    scores_list = []
    img_scores = {}
    for img in datapoint['imgs']:
        img_scores[img] = 0
        
    SKIP_DOC = 0
    for sent in sents[:30]:
        max_score = float('-inf')
        for img in datapoint['imgs']:
            if img != "":
                try:
                    cur_score = scorer.get_image_score(sent, img.split('.')[0]).item()
                    max_score = max(cur_score, max_score)
                    img_scores[img] += cur_score
                except:
                    continue
        
        if max_score < 0:
            SKIP_DOC = 1
        scores_list.append(str(max_score))
    
    if SKIP_DOC:
        continue

    top_ims = sorted(img_scores.items(), key=lambda item: item[1], reverse=True)[:3]
    top_ims = [x[0] for x in top_ims if x[0] != '']
    
    if 'imgs' in datapoint: del datapoint['imgs']
    if 'smp' in datapoint: del datapoint['smp']

    if len(top_ims) == 0:
        continue

    datapoint['imgs'] = top_ims
    datapoint['scores'] = scores_list
    dataset.append(datapoint)
    
    if cnt % 10000 == 0:
        with open('/scratch/full_data_test/test.{}.json'.format(FILE_NUM), 'w+') as f:
            json.dump(dataset, f)
    
    cnt += 1
    
with open('/scratch/full_data_test/test.{}.json'.format(FILE_NUM), 'w+') as f:
    json.dump(dataset, f)
