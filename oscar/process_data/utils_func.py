import pdb 
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt 
import clip
import pickle
import torch
import clip
import torch.nn.functional as F

# # prepare CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device) 

path = '../metrics/clip_self_retrieve/test_imgs_feats_5k.pkl'
with open(path, 'rb') as fin:
    save_datas = pickle.load(fin)
img_feats = save_datas['feats'] # [img num, 512]
img_names = save_datas['imgids']

name2id = {}
for i,img_name in enumerate(img_names):
    name2id[img_name] = i
img_num = len(img_names)

def get_text_feats(caps):
    # caps [N, length]
    text = clip.tokenize(caps).to(device) # [N, 77]
    # get text feature(dim=12)
    with torch.no_grad():
        text_features = model.encode_text(text).cpu()
        text_features = F.normalize(text_features.float())*10
    return text_features # [N, 512]

def get_img_feats(img_name):
    imgFeat_root = "../datasets/clip_feats/"
    for split in ['val','train','test']:
        img_id = img_name.replace('val', split)
        # print(imgFeat_root+img_id+'.npy')
        if img_id in img_names:
            idx = img_names.index(img_id)
            img_feat = img_feats[idx]
            return img_feat

def ClipRank(input_sent, imgid):
    # convert sent to list
    if isinstance(input_sent, str):
        sents = [input_sent]
    else:
        sents = input_sent
    scores = []
    for sent in sents:
        # get text clip feature
        text_feat = get_text_feats(sent) # [1,512]
        sim = torch.matmul(img_feats, text_feat.T) # [img num, 1]
        sim = sim.squeeze()
       
        sorted_idxs = torch.argsort(sim, descending=True).cpu().data.numpy().tolist()
        target_img_idx = name2id[imgid]
        init_rank = sorted_idxs.index(target_img_idx)
        clip_rank =  init_rank + 1
        scores.append(clip_rank)
    base = scores[0]
    scores = [base-s for s in scores]
    scores[0] = base
    return scores