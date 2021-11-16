from transformers.pytorch_transformers import BertModel, BertTokenizer, BertConfig
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import base64
from oscar.utils.tsv_file import TSVFile
import os
import os.path as op
import numpy as np

import torch.utils.data as DU
from oscar.modeling.modeling_bert import ImageBertForSequenceClassification
import json


class OscarScorer:
    def __init__(self, device):
        self.device = device 
        self.IMG_FEAT_DIR = '/scratch/summ_data_imgs_test'
        self.IMG2IDX_JSON = op.join(self.IMG_FEAT_DIR, 'img2idx.json')
        
        if not op.isfile(self.IMG2IDX_JSON):
            self.make_img2idx()
            
        with open(self.IMG2IDX_JSON, 'r') as f:
            print('img idx loaded')
            self.img2idx = json.load(f)
            
        self.features_file = TSVFile(os.path.join(self.IMG_FEAT_DIR, 'features.tsv'))
        self.labels_file = TSVFile(os.path.join(self.IMG_FEAT_DIR, 'labels.tsv')) 
        
        self.max_seq_len = 70
        self.max_img_seq_len = 70
        self.att_mask_type = 'CLR'
        
        self.oscar_config = BertConfig.from_pretrained('/home/anshul.padhi/msmo/models/oscar_ir')

        self.oscar_ir = ImageBertForSequenceClassification.from_pretrained('/home/anshul.padhi/msmo/models/oscar_ir', config=self.oscar_config).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # For oscar scores
        self.sm = nn.Softmax(dim=1)
            
    def make_img2idx(self):
        IMG_FILE = os.path.join(self.IMG_FEAT_DIR, 'features.tsv')
        LABEL_FILE = os.path.join(self.IMG_FEAT_DIR, 'labels.tsv')
        img2idx = {}
        print('making img2idx file...')
        with open(IMG_FILE, 'r') as img_file:
            with open(LABEL_FILE, 'r') as label_file:
                cnt = 0
                line = img_file.readline()
                line_label = label_file.readline()
                while line:
                    img_id = line.split('\t')[0].strip()
                    label_id = line.split('\t')[0].strip()

                    if img_id == label_id:
                        img2idx[img_id] = cnt
                        cnt += 1
                    else:
                        print('mismatch... How?')

                    line = img_file.readline()

        with open(self.IMG2IDX_JSON, 'w+') as f:
            json.dump(img2idx, f)

    def get_oscar_input(self, sent, img_id):
        idx = self.img2idx[img_id]
        features_row = self.features_file.seek(idx)
        labels_row = self.labels_file.seek(idx)
        
        # GET LABELS
        results = json.loads(labels_row[1].replace('\'', '\"'))
        objects = results['objects'] if type(
            results) == dict else results
        labels = {
            "image_h": results["image_h"] if type(
                results) == dict else 600,
            "image_w": results["image_w"] if type(
                results) == dict else 800,
            "class": [cur_d['class'] for cur_d in objects],
            "boxes": np.array([cur_d['rect'] for cur_d in objects],
                              dtype=np.float32)
        }
        
        od_labels = ' '.join(labels['class'])
     
        # GET IMAGE FEATURES
        num_boxes = int(features_row[1])
        features = np.frombuffer(base64.b64decode(features_row[-1]),
                                 dtype=np.float32).reshape((num_boxes, -1))
        t_features = torch.from_numpy(features)
        
        # TENSORIZE EXAMPLE
                
        cls_token_segment_id = 0
        pad_token_segment_id = 0
        sequence_a_segment_id = 0
        sequence_b_segment_id = 1
        
        text_a = sent
        text_b = od_labels
        img_feat = t_features
        
        tokens_a = self.tokenizer.tokenize(text_a)
        if len(tokens_a) > self.max_seq_len - 2:
            tokens_a = tokens_a[:(self.max_seq_len - 2)]

        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens_a) + 1)
        seq_a_len = len(tokens)
        if text_b:
            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > self.max_seq_len - len(tokens) - 1:
                tokens_b = tokens_b[: (self.max_seq_len - len(tokens) - 1)]
            tokens += tokens_b + [self.tokenizer.sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        seq_len = len(tokens)
        seq_padding_len = self.max_seq_len - seq_len
        tokens += [self.tokenizer.pad_token] * seq_padding_len
        segment_ids += [pad_token_segment_id] * seq_padding_len
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # image features
        img_len = img_feat.shape[0]
        if img_len > self.max_img_seq_len:
            img_feat = img_feat[0 : self.max_img_seq_len, :]
            img_len = img_feat.shape[0]
            img_padding_len = 0
        else:
            img_padding_len = self.max_img_seq_len - img_len
            padding_matrix = torch.zeros((img_padding_len, img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)

        # generate attention_mask
        att_mask_type = self.att_mask_type
        if att_mask_type == "CLR":
            attention_mask = [1] * seq_len + [0] * seq_padding_len + \
                             [1] * img_len + [0] * img_padding_len
            
        else:
            # use 2D mask to represent the attention
            max_len = self.max_seq_len + self.max_img_seq_len
            attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
            # full attention of C-C, L-L, R-R
            c_start, c_end = 0, seq_a_len
            l_start, l_end = seq_a_len, seq_len
            r_start, r_end = self.max_seq_len, self.max_seq_len + img_len
            attention_mask[c_start : c_end, c_start : c_end] = 1
            attention_mask[l_start : l_end, l_start : l_end] = 1
            attention_mask[r_start : r_end, r_start : r_end] = 1
            if att_mask_type == 'CL':
                attention_mask[c_start : c_end, l_start : l_end] = 1
                attention_mask[l_start : l_end, c_start : c_end] = 1
            elif att_mask_type == 'CR':
                attention_mask[c_start : c_end, r_start : r_end] = 1
                attention_mask[r_start : r_end, c_start : c_end] = 1
            elif att_mask_type == 'LR':
                attention_mask[l_start : l_end, r_start : r_end] = 1
                attention_mask[r_start : r_end, l_start : l_end] = 1
            else:
                raise ValueError("Unsupported attention mask type {}".format(att_mask_type))
        
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0).to(self.device)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        img_feat = img_feat.unsqueeze(0).to(self.device)
        
        label = 1
        #return idx, tuple([input_ids, attention_mask, segment_ids, img_feat, label])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': segment_ids,
            'img_feats': img_feat,
            'labels': None
        }
        
    def _get_image_score(self, inp):
        logits = self.oscar_ir(**inp)[:2][0]
        probs = self.sm(logits)
        result = probs[:, 1]
        
        return result
    
    def get_image_score(self, sent, img_id):
        inp = self.get_oscar_input(sent, img_id)
        score = self._get_image_score(inp)
        return score
