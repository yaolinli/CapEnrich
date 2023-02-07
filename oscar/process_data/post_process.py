import torch
from PIL import Image
import torch.nn.functional as F
import os
import pdb 
import numpy as np
from tqdm import tqdm
import argparse
from numpy import dot
from numpy.linalg import norm
import json
import pickle
import matplotlib.pyplot as plt  
import spacy
from utils_func import get_text_feats, get_img_feats, ClipRank
import sys
nlp = spacy.load("en_core_web_sm")

IMAGE_ROOT = '../datasets/images/' 

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def integrity(sent):
    if '.' not in sent:
        sent = sent + '.'

    sent = sent.split(', ')[-1]
    sent = sent.replace('.', '')
    if len(sent) > 1:
        doc = nlp(sent)
        root = [token for token in doc if token.head == token][0]
        subject = list(root.rights)
        if len(subject) > 0:
            return True
    return False

def readline_json(file):
    datas = {}
    with open(file, 'r', encoding='utf-8') as fin:
        for line in fin:
            row = json.loads(line)
            image_id = row['image_id']
            caption = row['caption']
            conf = row['conf']
            if image_id not in datas.keys():
                datas[image_id] = []
            datas[image_id].append({'caption':caption, 'conf':conf})
    return datas

def isin(sent, prompts):
    sent = sent.replace(',', '.')
    if (sent in prompts) or (sent[:-1] in prompts):
        return True
    else:
        return False
    
def process(cap):
    cap = cap.replace('x ', '')
    cap = cap.replace(' x', '')
    cap = cap.replace(' x x', '')
    cap = cap.replace('x x ', '')
    cap = cap.replace('x x', '')
    if '.' not in cap:
        cap = cap + '.'
    new = cap.replace('.', ',')
    if new[-1] == ',':
        new = new[:-1] + '.'
    new = new.replace(',,', ',')
    new = new.replace(',.', '.')
    return new

if __name__=="__main__":
    result_path = '../eval_results/output_3e-4_nctx2_random.json'
    vinvl_results = read_json('../eval_results/vinvl_result.json')
    cur_datas = readline_json(result_path)

    id = 0
    save_datas = []
    noprmpt_count = 0
    nobeat_count = 0
    nobeat_imgids = []
    for i,jterm in tqdm(enumerate(vinvl_results)):
        imgid = jterm["image_id"]
        general_cap = jterm["caption"]
        if imgid not in cur_datas.keys():
            data = {}
            data['image_id'] = imgid
            data['id'] = id
            data['caption'] = general_cap
            id += 1
            save_datas.append(data)
            noprmpt_count += 1
            continue
        
        endprmpt_cap = []
        for cap in cur_datas[imgid]:
            new = process(cap['caption'])
            if len(new) <= 1:
                continue
            if integrity(new):
                endprmpt_cap.append(new) 
                
        if len(endprmpt_cap) == 0:
            data = {}
            data['image_id'] = imgid
            data['id'] = id
            data['caption'] = general_cap
            id += 1
            save_datas.append(data)
            nobeat_count += 1
            nobeat_imgids.append(imgid)
            continue
        
        prmpt_base = False

        all_caps = [general_cap] + endprmpt_cap
        text_features = get_text_feats(all_caps)
        img_name = 'COCO_val2014_'+'0'*(12-len(imgid))+imgid
        img_feature = get_img_feats(img_name)
        torch_sim_scores = torch.matmul(img_feature, text_features.T).squeeze()
        # calculate clip match score
        sim_scores = torch_sim_scores.data.numpy()
        base_score = sim_scores[0]
        endprmpt_scores = sim_scores[1:]
        sort_scores, sort_idxs = torch.sort(torch_sim_scores[1:])
        max_score = sort_scores[-1]
        max_ci = sort_idxs[-1]
        # calculate relative rank
        ranks = ClipRank(all_caps, img_name)
        max_rank = max(ranks)
        
        if (max_score > base_score) or  (ranks[max_ci] > ranks[0]): # clip score > base
            data = {}
            data['image_id'] = imgid
            data['id'] = id
            data['caption'] = endprmpt_cap[max_ci]
            id += 1
            save_datas.append(data)
            prmpt_base = True

        else:
            data = {}
            data['image_id'] = imgid
            data['id'] = id
            data['caption'] = general_cap
            id += 1
            save_datas.append(data)
            nobeat_count += 1
            nobeat_imgids.append(imgid)
    count = nobeat_count + noprmpt_count


    out_path = result_path.split('.json')[0] + '_eval.json'
    with open(out_path, 'w') as fout:
        json.dump(save_datas, fout)
