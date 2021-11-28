# Loader for the MSMO dataset
# Requires preprocessing to include the simplified sentences and images list

import pandas as pd 
import cv2
from ast import literal_eval
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import random

class MSMO_Data(Dataset):
    def __init__(self, csv_file, root_dir='/scratch/anshul/valid_data/', transform=None):
        self.transform = transform
        self.df = pd.read_csv(root_dir+csv_file)
        self.root_dir = root_dir
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        article_file = self.root_dir+'article/'+self.df['article'][idx]
        art = open(article_file)
        art = art.read()
        art = art.replace("\n", " ")
        art = art.rstrip()
        text = art.split("@title ")[1].split("@summary ")[0]

        summaries = art.split("@title ")[1].split("@summary ")[1:]
        summary = ""
        for i in summaries:
            i = i.rstrip()
            summary+=i
            summary+=". "
        

        imgs = []
        x = literal_eval(self.df['img'][idx])
        k=len(x)
        arr = [i for i in range(k)]
        if len(x)>15:
            k=15
            arr = random.sample(range(len(x)), 15)
        for i in arr:
            img_name = self.root_dir+'img/'+x[i]
            im = cv2.imread(img_name)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (150, 150))
            if(self.transform):
                im = self.transform(im)
            imgs.append(im)
        for i in range(15-k):
            imgs.append(np.zeros((150, 150, 3)))
        
        data_pt = {'text':text, 'text_simple':simple, 'summary':summary, 'im_list':imgs}
        return data_pt
        
        

