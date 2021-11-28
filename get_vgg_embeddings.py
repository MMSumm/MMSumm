# File to produce VGG-19 embeddings
# Provide path to images

from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
from keras.models import Model

import numpy as np
from tqdm import tqdm_notebook as tqdm
import re
import matplotlib.pyplot as plt
import string
import os
from IPython.display import Image
import csv
import base64
from tqdm import tqdm


model = VGG19()
model = Model(inputs=model.input, outputs=model.get_layer('fc2').output)

def get_image_embed(fname):
    img = image.load_img(fname, target_size=(224, 224))
    img_data = np.expand_dims(image.img_to_array(img), axis=0)
    img_data = preprocess_input(img_data)
    embed = model.predict(img_data)
    return embed


prefix = '/scratch/full_data_test/test_data/img/'
imgs = os.scandir(prefix)

feature_file_name = '/scratch/summ_data_imgs_test/vgg_features.tsv'

FIELDNAMES = ['img', 'features']


with open(feature_file_name, 'a+') as feature_file:
    writer = csv.DictWriter(feature_file, delimiter='\t', fieldnames=FIELDNAMES)

    for img in tqdm(imgs):
        try:
            feats = get_image_embed(os.path.join(prefix, img.name))
            feats_encoded = base64.b64encode(feats).decode('utf-8')
            item = {
                'img': img.name,
                'features': feats_encoded
            }
            writer.writerow(item)
        except Exception as e:
            if type(e) is KeyboardInterrupt:
                break
            else:
                continue
