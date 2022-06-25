import torch
import clip
from PIL import Image
import os
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
probs = []
for i in os.listdir('data'):
    text_dir = i+'/article/'
    img_dir = i+'/images/'
    text_file = text_dir+'article.txt'
    text = clip.tokenize(open(text_file).read()).to(device)
    text_features = model.encode_text(text)
    prob = []
    for j in os.listdir(img_dir)
        image = preprocess(Image.open(j)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            logits_per_image, logits_per_text = model(image, text)
            prob.append(logits_per_image.softmax(dim=-1).cpu().numpy())
    probs.append(np.array(prob))
probs = np.array(probs)

np.savetxt('clip.out', x, delimiter=',')
