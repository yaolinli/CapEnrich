import torch
import clip
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
import clip

# prepare CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
# device =  "cpu"
model, preprocess = clip.load("ViT-B/32", device=device) 

def get_text_feat(caps):
    # caps [N, length]
    text = clip.tokenize(caps).to(device) # [N, 77]
    # get text feature(dim=12)
    with torch.no_grad():
        text_features = model.encode_text(text).cpu()
        text_features = F.normalize(text_features.float())*10
    return text_features

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--testfile', type=str, default='../../eval_results/vinvl_result.json') 
    parser.add_argument('--retrieve_set', type=str, default='hard') # or 'naive'
    parser.add_argument('--split', type=str, default='test') 
    args = parser.parse_args()
    
    if '.json' in args.testfile:
        files = [args.testfile]
    else:
        files = os.listdir(args.testfile)
        files = [os.path.join(args.testfile, f) for f in files]
        
    for file in files:
        if ('.json' in file):
            print("=================================================")
            print(file)
            print("=================================================")
            gcaps = read_json(file)
            text_dict = {}
            for gcap in gcaps:
                caption = gcap["caption"].split("x ")[0]
                text_dict['COCO_val2014_'+'0'*(12-len(gcap["image_id"]))+gcap["image_id"]] = caption
            print("generated caption size", len(text_dict))

            # get split ids
            if args.split == 'test':
                test_ids = np.load('../../datasets/annotation/tst_names.npy')
            else:
                test_ids = np.load('../../datasets/annotation/val_names.npy')

            test_ids = [id.split('/')[-1][:-4] for id in test_ids]
            
            
            text_feats = {}
            # ids = list(text_dict.keys())
            ids = list(text_dict.keys())
            for img_id in tqdm(ids):
                text_feats[img_id] = get_text_feat(text_dict[img_id]) # [1, 512]
            
            if args.retrieve_set == 'hard':
                saved_clip_feats = './test_imgs_feats_3w.pkl'
            else: # naive
                saved_clip_feats = './test_imgs_feats_5k.pkl'
            with open(saved_clip_feats, 'rb') as fin:
                datas = pickle.load(fin)
            all_img_feats = datas['feats'].squeeze()
            img_names = datas['imgids']

            print("retrieve image set {}".format(len(img_names)))
            
            # retrieve similar images based in clip features
            K = 30   # get top k similar images
            groups = {}
            ref_ids = img_names
            recall = {1:0,5:0,10:0}
            for i, img_id in tqdm(enumerate(ids)):

                groups[img_id] = {}
                cur_txt_feats = text_feats[img_id]
                scores_arry = all_img_feats.matmul(cur_txt_feats.T) # [N, 5]
                scores = torch.sum(scores_arry, dim=-1) # [N,]
                values, indexs = torch.topk(scores, K)
                groups[img_id]['ids'] = [ref_ids[idx] for idx in indexs]
                groups[img_id]['scores'] = values
                for k in [1,5,10]:
                    if img_id in groups[img_id]['ids'][:k]:
                        recall[k] += 1

            
            for k in [1,5,10]:
                recall[k] /= len(ids)
                print("R@{}:{:3}".format(k, recall[k]))

                    
                
                
                

        