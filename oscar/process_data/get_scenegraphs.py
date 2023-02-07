import json
import matplotlib.pyplot as plt  
import os
import numpy as np
import sng_parser
from pprint import pprint
from tqdm import tqdm 

def read_json(file):
    datas = []
    with open(file, 'r', encoding='utf-8') as fin:
        datas = json.load(fin)
    return datas

def convert_sg_format(raw_sg):
    sg = {}
    sg['objects'] = []
    sg['spans'] = []
    sg['relations'] = []
    sg['attributes'] = {}
    for term in raw_sg['entities']:
        sg['objects'].append(term['lemma_head'])
        sg['spans'].append(term['span'])
        if len(term['modifiers']) > 0:
            for mod in term['modifiers']:
                if mod['dep'] != 'det':
                    if term['lemma_head'] not in sg['attributes'].keys():
                        sg['attributes'][term['lemma_head']] = []
                    sg['attributes'][term['lemma_head']].append([mod['span'],mod['dep']])
    for term in raw_sg['relations']:
        sg['relations'].append((sg['objects'][term['subject']],term['relation'],sg['objects'][term['object']]))
    return sg 

def scene_graph(sentence):
    # print('Sentence:', sentence)
    # Here we just use the default parser.
    result = sng_parser.parse(sentence)
    # print(result)
    # sng_parser.tprint(result)
    # sng_parser.tprint(sng_parser.parse(sentence), show_entities=False)
    graph = sng_parser.parse(sentence)
    #pprint(graph)
    sg = convert_sg_format(result)
    return sg

if __name__=="__main__":
    # read caption json
    cap_path = '../datasets/annotation/ref_captions.json'
    with open(cap_path, 'r', encoding='utf-8') as fin:
        all_captions = json.load(fin)
    IMAGE_ROOT = '../datasets/images/'  
    # get split image ids
    split_root = '../datasets/annotation/'
    img_ids = []
    for split in ['trn','val','tst']:
        split_path = split_root + '{}_names.npy'.format(split)
        img_ids.extend(np.load(split_path))
    # get caption set
    captions = {}
    id2name = {}
    for k in img_ids:
        new_k = str(int(k.split('_')[-1].split('.jpg')[0]))
        captions[new_k] = all_captions[k]
        id2name[new_k] = k

    '''
    sample in scene_graphs as follows:
    key: image id 487008 
    value:
    {
        'a room with a couch chair and television':
        {
            'objects': ['room', 'chair', 'television'],
            'relations': [('room', 'with', 'television'), ('room', 'with', 'chair')],
            'attributes': {'room': [], 'chair': ['couch'], 'television': []}, 
        }
    }
    '''
    scene_graphs = {}
    for k,v in tqdm(captions.items()):
        scene_graphs[k] = {}
        for cap in v:
            scene_graphs[k][cap] = scene_graph(cap)

    out_path = '../datasets/ref_cap2sng.json'
    with open(out_path, 'w') as fout:
        json.dump(scene_graphs, fout)