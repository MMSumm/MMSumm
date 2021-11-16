#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
from keras.models import Model

import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
import re
import matplotlib.pyplot as plt
import string
import os
from IPython.display import Image


# In[2]:


import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


# In[3]:


trainfile = './mediaeval2016/train_posts.txt'
testfile = './mediaeval2016/test_posts.txt'

trainimgdir = './mediaeval2016/devset/Mediaeval2016_DevSet_Images/'
testimgdir = './mediaeval2016/testset/Mediaeval2016_TestSet_Images/'

train = pd.read_csv(trainfile, sep='\t')
test = pd.read_csv(testfile, sep='\t')


# In[4]:


def return_first_image(row):
    missing = set(['boston_fake_35',
                     'eclipse_video_01',
                     'sandy_real_09',
                     'sandy_real_10',
                     'sandy_real_4',
                     'sandy_real_6',
                     'sandy_real_90',
                     'syrianboy_1',
                     'varoufakis_1'])
    
    for img in row['image_id(s)'].split(','):
        return img.strip()


def retrieve_file_name(row,dirname):
    filenames = {i.split('.')[0].strip():i for i in os.listdir(dirname)}
    try:
        if filenames[row['first_image_id']].endswith('txt'):
            return float('nan')
        else:
            return dirname+filenames[row['first_image_id']]
    except:
        return float('nan')

train['first_image_id'] = train.apply (lambda row: return_first_image(row),axis=1)
train['image_filename'] = train.apply (lambda row: retrieve_file_name(row,trainimgdir),axis=1)


# In[5]:


train = train.dropna()


# In[6]:


test['first_image_id'] = test['image_id']
test['image_filename'] = test.apply (lambda row: retrieve_file_name(row,testimgdir),axis=1)


# In[7]:


test = test.dropna()


# In[8]:


images_train_dataset = [i for i in train['first_image_id'].tolist()]
images_train_folder = [i.split('.')[0].strip() for i in os.listdir(trainimgdir)]
images_train_not_available = set(images_train_dataset)-set(images_train_folder)


# In[9]:


class mediaevalDataset(Dataset):
    def get_image(self,filename,traindataflag):
        return io.imread(filename)
#         if traindataflag:
#             return io.imread(self.trainimgdir+filename)
#         else:
#             return io.imread(self.testimgdir+filename)

    def __init__(self,df,traindataflag):
        self.data = df
        self.traindataflag = traindataflag
        self.trainimgdir = './mediaeval2016/devset/Medieval2016_DevSet_Images/'
        self.testimgdir = './mediaeval2016/testset/Medieval2016_TestSet_Images/'
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        img_name = self.data.loc[idx]['image_filename']
        image = self.get_image(img_name,self.traindataflag)
        text = self.data.loc[idx]['post_text']
        post_id = self.data.loc[idx]['post_id']
    
        sample = {'image': image, 'text': text, 'post_id':post_id}
        
        return sample


# In[10]:


td = mediaevalDataset(train,1)
test_d = mediaevalDataset(test, 0)
tdl = DataLoader(dataset=td, batch_size=1, shuffle=True)


# In[14]:


model = VGG19()
model = Model(inputs=model.input, outputs=model.get_layer('predictions').output)

image_embed = {}

for it in test_d.data.itertuples():
    fname = it.image_filename
    if fname not in image_embed:
        try:
            img = image.load_img(fname, target_size=(224, 224))
            img_data = np.expand_dims(image.img_to_array(img), axis=0)
            img_data = preprocess_input(img_data)
            image_embed[fname] = model.predict(img_data)
        except:
            continue
    else:
        continue

for it in td.data.itertuples():
    fname = it.image_filename
    if fname not in image_embed:
        try:
            img = image.load_img(fname, target_size=(224, 224))
            img_data = np.expand_dims(image.img_to_array(img), axis=0)
            img_data = preprocess_input(img_data)
            image_embed[fname] = model.predict(img_data)
        except:
            continue
    else:
        continue
        
import pickle
pickle.dump(image_embed, open("./image_embed.pkl", "wb+"))
